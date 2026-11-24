"""
Built-in template: nested_miller_3stage_az_cascoded_input_mirror.

Yan AZ 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** following the R85 C cross-variant
cascode recipe. R87 C batch rollout #3 — applies the R85 C cascode
recipe (telescopic PMOS input cascode + DFCFC1-style regulated NMOS
mirror with diode loop at high-Z node ``d1``) to AZ's auto-zero +
nulling-R + active-FF backbone.

Why this variant
================

AZ has plain NMCF stage 1 (PMOS diff pair + NMOS diode mirror) -
R85 C cascode recipe drops in verbatim. AZ's distinguishing
compensation:

  1. Auto-zero NMOS pair: Mgmb1 (d=naz_drv, g=naz_cap) +
     Mgmb2 (d=naz_pair, g=naz_drv). Reciprocal gate/drain coupling.
  2. Active FF Mgmf2 (PMOS, g=d2, d=out) - LHP zero via FF leg.
  3. Single nulling-R Raz + single auto-zero cap Caz in series.
  4. Outer Miller Caz_outer (out<->d2) NMCF-style preserved.

AZ paper GBW = 5 MHz (from AnalogGym GraphAMPAZ.GBW_target).
Initial Caz_outer ~1 pF; Mgmf2_scale = 0.1 (DC-preserving).

Architectural devices preserved verbatim
=========================================

AZ fingerprint that survives R85 C cascode swap:

  1. **Mgmb1** (NMOS, d=naz_drv, g=naz_cap, s=vss) - auto-zero
     sensing device.
  2. **Mgmb2** (NMOS, d=naz_pair, g=naz_drv, s=vss) - auto-zero
     mirror device.
  3. **Mgmf2** (PMOS, g=d2, d=out, s=vdd) - active FF.
  4. **Raz** (resistor, naz_drv <-> naz_cap) - primary nulling R.
  5. **Caz** (cap, naz_cap <-> naz_pair) - auto-zero series cap.
  6. **Caz_outer** (cap, out <-> d2) - NMCF-style outer Miller.
  7. **DC bias paths** Mbn_naz_pair + Mbp_naz_cap.

R85 C cascode recipe applied
============================

  * Mcas1 / Mcas2 PMOS cascode pair on input.
  * Mcas_b + Iref_cas - pcas rail bias.
  * Mcas3_n / Mcas4_n NMOS cascode pair on mirror legs. M3/M4
    gates closed at ``d1_diode``.
  * Mcas_n_b1 + Mcas_n_b2 + Mbp_ncas_drv ncas rail.

Honest scope
============

AZ's auto-zero loop is a charge-redistribution structure designed
to null DC offset; cascoded mirror adds +15-20 dB DC gain.
Expected gain_dc ~120-128 dB. All seven W's BYTE-EQUAL with R86 D
v124 / R85 C v123.
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


def build_nested_miller_3stage_az_cascoded_input_mirror(
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

    # AZ-specific: auto-zero NMOS pair sizing
    W_gmb: float = 4e-6,
    L_gmb: float = 1e-6,

    # AZ-specific: active FF Mgmf2
    Mgmf2_scale: float = 0.1,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # AZ-specific: compensation network
    Caz_outer: float = 1.0e-12,  # outer Miller (out<->d2)
    Caz:       float = 0.5e-12,  # auto-zero cap (naz_cap<->naz_pair)
    Raz:       float = 10e3,     # primary nulling-R (naz_drv<->naz_cap)

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
    # cascoded AZ 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     directly to input-referred offset; the AZ auto-zero loop
    #     cancels DC offset over the chopper-like cycle but the
    #     short-term mismatch still seeds settling error.
    #   * MATCHING on the PMOS input cascode pair (Mcas1, Mcas2). The
    #     telescopic cascode lifts stage-1 output impedance only when
    #     Mcas1/Mcas2 share Vds bias; mismatch unbalances the cascode
    #     and re-introduces offset at d1.
    #   * MATCHING on the AZ NMOS pair (Mgmb1, Mgmb2). Reciprocal
    #     gate/drain coupling (Mgmb1 g=naz_cap d=naz_drv;
    #     Mgmb2 g=naz_drv d=naz_pair) makes the AZ loop sensitive to
    #     Vth mismatch — a mismatched pair would skew the auto-zero
    #     reference and leak residual DC offset to out.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Yan AZ 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.
    AZ signature (auto-zero NMOS pair + nulling R + active FF)
    preserved verbatim.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # AZ: Mgmf2 draws additional DC current at out (same as AFFC).
    I_gmf2 = Mgmf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(K_cs3_load * (1 + Mgmf2_scale)))
    if K_cs3_load_total < 1:
        K_cs3_load_total = 1

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

    # Mgmf2 sizing.
    if L_gmf2 is None:
        L_gmf2 = L_cs3
    if gm_id_gmf2 is not None:
        W_gmf2 = closed_form_W("pmos", gm_id=gm_id_gmf2,
                               I_d=I_gmf2, L=L_gmf2).W
    elif W_gmf2 is None:
        W_gmf2 = W_cs3 * Mgmf2_scale

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

    # AZ delta: auto-zero NMOS pair
    devices.append(_mosfet("Mgmb1", "nmos",
                            d="naz_drv", g="naz_cap", s="vss", b="vss",
                            W=W_gmb, L=L_gmb))
    devices.append(_mosfet("Mgmb2", "nmos",
                            d="naz_pair", g="naz_drv", s="vss", b="vss",
                            W=W_gmb, L=L_gmb))

    # AZ delta: active FF Mgmf2 (PMOS, g=d2, d=out)
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # AZ compensation: Caz_outer (NMCF-style outer Miller) + Caz +
    # Raz (auto-zero series).
    devices.append(_cap("Caz_outer", p="out", n="d2", value=Caz_outer))
    devices.append(_cap("Caz", p="naz_cap", n="naz_pair", value=Caz))
    devices.append(_res("Raz", p="naz_drv", n="naz_cap", value=Raz))

    # DC bias paths for AZ internal nodes.
    devices.append(_mosfet("Mbn_naz_pair", "nmos",
                            d="naz_pair", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbp_naz_cap", "pmos",
                            d="naz_cap", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R117 C: optional DRC annotations
    # Three MATCHING annotations: PMOS diff pair, PMOS input cascode
    # pair, and AZ NMOS pair. See the kwarg docs block above for the
    # per-annotation rationale.
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
                    "input-referred offset. AZ auto-zero cancels DC "
                    "offset over the chopper-like cycle but short-term "
                    "mismatch still seeds settling error at d1"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mcas1", "Mcas2"),
            extras={
                "rationale": (
                    "PMOS input cascode pair (R85 C recipe) — "
                    "telescopic cascode lifts stage-1 output impedance "
                    "only when Mcas1/Mcas2 share Vds bias; mismatch "
                    "unbalances the cascode and re-introduces offset "
                    "at d1, partially cancelling the cascode gain "
                    "product the AZ loop relies on"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmb1", "Mgmb2"),
            extras={
                "rationale": (
                    "AZ NMOS pair with reciprocal gate/drain coupling "
                    "(Mgmb1 g=naz_cap d=naz_drv; Mgmb2 g=naz_drv "
                    "d=naz_pair) — Vth mismatch skews the auto-zero "
                    "reference and leaks residual DC offset to out"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
