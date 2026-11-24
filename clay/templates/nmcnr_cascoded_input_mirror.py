"""
Built-in template: nested_miller_3stage_nmcnr_cascoded_input_mirror.

Leung NMCNR 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** following the R85 C cross-variant
cascode recipe. R87 C batch rollout #2 — applies the R85 C cascode
recipe (telescopic PMOS input cascode + DFCFC1-style regulated NMOS
mirror with diode loop at high-Z node ``d1``) to NMCNR's nulling-
resistor compensation backbone.

Why this variant
================

NMCNR has plain NMCF stage 1 (PMOS diff pair + NMOS diode mirror) -
R85 C cascode recipe drops in verbatim with NO topology re-engineering.
NMCNR's distinguishing compensation is the **nulling resistor Rz1**
+ shared **rz_n node** + two Miller caps (Cm1 = rz_n<->d2 outer,
Cm2 = rz_n<->d1 inner). Cf is dropped — the nulling resistor handles
RHP-zero cancellation.

NMCNR paper GBW = 1.2 MHz (from AnalogGym GraphAMPNMCNR.GBW_target).
Initial Cm1 estimate ~4 pF for 1.2 MHz target.

Architectural devices preserved verbatim
=========================================

NMCNR fingerprint that survives R85 C cascode swap:

  1. **rz_n node** + **Rz1 resistor** (rz_n<->out) - the nulling-R
     signature.
  2. **Cm1** (rz_n<->d2) outer Miller terminating at rz_n.
  3. **Cm2** (rz_n<->d1) inner Miller ALSO terminating at rz_n.
  4. NO Cf.

R85 C cascode recipe applied
============================

  * Mcas1 / Mcas2 PMOS cascode pair on input.
  * Mcas_b + Iref_cas - pcas rail bias.
  * Mcas3_n / Mcas4_n NMOS cascode pair on mirror legs. M3/M4
    gates closed at ``d1_diode`` (DFCFC1-style regulated cascode
    loop).
  * Mcas_n_b1 + Mcas_n_b2 stacked NMOS-diode pair for ncas rail.
  * Mbp_ncas_drv PMOS mirror at pbias sources I_ref.

Honest scope
============

NMCNR's nulling-R Rz1 is a textbook RHP-zero migration; cascoded
mirror adds +15-20 dB DC gain via cascode product. Expected
gain_dc ~120-128 dB; GBW retune required as cascoded mirror moves
the dominant pole. All seven W's BYTE-EQUAL with R86 D v124 /
R85 C v123.
"""

from __future__ import annotations

from ..ir import Cell, Device, Spec


def _mosfet(name: str, kind: str, d: str, g: str, s: str, b: str,
            W: float, L: float, M: int = 1) -> Device:
    return Device(
        instance_name=name,
        device_class=kind,
        pdk_variant=None,
        pins={"d": d, "g": g, "s": s, "b": b},
        sizing={"W": W, "L": L, "M": float(M)},
    )


def _cap(name: str, p: str, n: str, value: float) -> Device:
    return Device(
        instance_name=name,
        device_class="cap",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def _res(name: str, p: str, n: str, value: float) -> Device:
    return Device(
        instance_name=name,
        device_class="res",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def build_nested_miller_3stage_nmcnr_cascoded_input_mirror(
    cell_name: str,
    spec: Spec,
    *,
    # Bias / mirror ratios
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 4,

    # Per-device sizing: physical (W, L) form
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 40e-6,   L_cs3: float = 0.5e-6,

    # Per-device sizing: intent (gm/Id) form
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # NMCNR compensation: TWO Miller caps + nulling R
    Cm1: float = 4.0e-12,   # outer Miller (rz_n <-> d2)
    Cm2: float = 0.5e-12,   # inner Miller (rz_n <-> d1)
    Rz1: float = 5e3,       # nulling resistor (rz_n <-> out)

    # R83 D inherited: PMOS cascode pair on the input
    W_cas: float = 20e-6,
    L_cas: float = 0.5e-6,
    I_ref_cas: float = 1e-6,

    # R84 D inherited: NMOS cascode pair on mirror
    W_cas_n_mirror: float = 4e-6,
    L_cas_n_mirror: float = 1e-6,
    W_cas_n_bias: float = 4e-6,
    L_cas_n_bias: float = 1e-6,
    I_ref_cas_n: float = 1e-5,

    # R117 C: optional qualitative DRC annotations
    # When True, attach three R111 C-vocabulary annotations on the
    # cascoded NMCNR 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     directly to input-referred offset; common-centroid required.
    #   * PASSIVE_MATCHING on the NMCNR Miller cap pair (Cm1, Cm2).
    #     Both share the rz_n shared node — Cm1 routes rz_n↔d2 (outer),
    #     Cm2 routes rz_n↔d1 (inner). The Rz1 nulling resistor's RHP-
    #     zero migration depends on the Cm1/Cm2 ratio being preserved
    #     over process; fringing-cap unit arrays make the two caps
    #     track on the same gradient.
    #   * MATCHING on the NMOS mirror cascode pair (Mcas3_n, Mcas4_n).
    #     The R84 D regulated cascode loop closes via M3/M4 gates at
    #     d1_diode; mismatched Mcas3_n/Mcas4_n unbalances the cascode
    #     output impedance and degrades the +15-20 dB gain boost.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Leung NMCNR 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.
    NMCNR signature (rz_n shared node + Rz1 nulling resistor + 2
    Miller caps to rz_n) preserved verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_load1, L=L_load1).W
    if gm_id_stage2 is not None:
        W_cs2 = closed_form_W("nmos", gm_id=gm_id_stage2,
                              I_d=I_cs2, L=L_cs2).W
    if gm_id_stage3 is not None:
        W_cs3 = closed_form_W("pmos", gm_id=gm_id_stage3,
                              I_d=I_cs3, L=L_cs3).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain
    devices.append(_mosfet("Mbp",  "pmos", d="pbias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn",  "nmos", d="nbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbnm", "nmos", d="pbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    # Cascode bias rail (pcas)
    devices.append(_mosfet("Mcas_b", "pmos",
                            d="pcas", g="pcas", s="vdd", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(Device(
        instance_name="Iref_cas",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "pcas", "n": "vss"},
        sizing={"value": I_ref_cas},
    ))

    # NMOS mirror-cascode bias rail (ncas)
    devices.append(_mosfet("Mcas_n_b1", "nmos",
                            d="ncas_int", g="ncas_int",
                            s="vss", b="vss",
                            W=W_cas_n_bias, L=L_cas_n_bias))
    devices.append(_mosfet("Mcas_n_b2", "nmos",
                            d="ncas", g="ncas",
                            s="ncas_int", b="vss",
                            W=W_cas_n_bias, L=L_cas_n_bias))
    devices.append(_mosfet("Mbp_ncas_drv", "pmos",
                            d="ncas", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Stage 1: cascoded PMOS diff pair with cascoded NMOS mirror
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode_int", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1_int",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    devices.append(_mosfet("Mcas1", "pmos",
                            d="d1_diode", g="pcas", s="d1_diode_int", b="vdd",
                            W=W_cas, L=L_cas))
    devices.append(_mosfet("Mcas2", "pmos",
                            d="d1",       g="pcas", s="d1_int",       b="vdd",
                            W=W_cas, L=L_cas))

    devices.append(_mosfet("M3", "nmos",
                            d="d1_diode_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                            d="d1_n_int", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    devices.append(_mosfet("Mcas3_n", "nmos",
                            d="d1_diode", g="ncas", s="d1_diode_n_int", b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))
    devices.append(_mosfet("Mcas4_n", "nmos",
                            d="d1",       g="ncas", s="d1_n_int",       b="vss",
                            W=W_cas_n_mirror, L=L_cas_n_mirror))

    # Stage 2: NMOS CS amplifier driven by d1
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # Stage 2 replica-bias leg
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Stage 3: PMOS CS amplifier (output stage)
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # Stage-3 NMOS sink (unchanged from NMCF baseline).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # NMCNR compensation: nulling resistor + 2 Miller caps to rz_n
    devices.append(_cap("Cm1", p="rz_n", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="rz_n", n="d1", value=Cm2))
    devices.append(_res("Rz1", p="rz_n", n="out", value=Rz1))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R117 C: optional DRC annotations
    # MATCHING (diff pair) / PASSIVE_MATCHING (Miller cap pair to rz_n)
    # / MATCHING (NMOS mirror cascode pair). See kwarg docs above.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": (
                    "PMOS diff pair — Vth mismatch maps directly to "
                    "input-referred offset; common-centroid required. "
                    "Stage 1 sits behind a telescopic input cascode "
                    "(Mcas1/Mcas2) and a regulated NMOS mirror cascode, "
                    "so M1/M2 imbalance is the dominant residual offset "
                    "source"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "NMCNR Miller cap pair — both share the rz_n node "
                    "(Cm1: rz_n↔d2 outer; Cm2: rz_n↔d1 inner). The Rz1 "
                    "nulling resistor's RHP-zero migration depends on "
                    "the Cm1/Cm2 ratio being preserved over process; "
                    "fringing-cap unit arrays make the two caps track "
                    "on the same gradient"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas3_n", "Mcas4_n"),
            extras={
                "rationale": (
                    "NMOS mirror cascode pair (R84 D regulated cascode) "
                    "— gates closed at d1_diode (DFCFC1-style diode "
                    "loop). Mismatched Mcas3_n/Mcas4_n unbalances the "
                    "cascode output impedance and degrades the "
                    "+15-20 dB gain boost from the cascoded mirror"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
