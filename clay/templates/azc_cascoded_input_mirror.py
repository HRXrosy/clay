"""
Built-in template: nested_miller_3stage_azc_cascoded_input_mirror.

Qu2017 AZC 3-stage opamp with **BOTH a cascoded PMOS input pair AND
a cascoded NMOS mirror load** following the R85 C cross-variant
cascode recipe. R87 C batch rollout #4 — applies the R85 C cascode
recipe (telescopic PMOS input cascode + DFCFC1-style regulated NMOS
mirror with diode loop at high-Z node ``d1``) to AZC's charge-
redistribution archetype trio + quad-R mesh + triple-cap backbone.

Why this variant
================

AZC has plain NMCF stage 1 (PMOS diff pair + NMOS diode mirror) -
R85 C cascode recipe drops in verbatim. AZC's distinguishing
compensation:

  1. Triple-MOSFET archetype: Mazc1 (NMOS, g=d2), Mazc2_diode (PMOS
     diode at nazc1), Mazc3 (NMOS, g=nazc_cap2).
  2. Quad-R nulling-resistor mesh: Razc0-Razc3 with two shared hub
     nodes (nazc_bias, nazc_hub).
  3. Triple-cap network: Cazc0 (out<->nazc_bias outer Miller-like),
     Cazc1 (nazc_cap1<->vss), Cazc2 (nazc_cap2<->vss). Inner caps
     to vss is AZC signature.

AZC paper GBW = 3 MHz (from AnalogGym GraphAMPAZC.GBW_target).
Initial Cazc0 ~1.5 pF for 3 MHz target.

Architectural devices preserved verbatim
=========================================

AZC fingerprint that survives R85 C cascode swap:

  1. **Mazc1** (NMOS, d=nazc1, g=d2, s=vss).
  2. **Mazc2_diode** (PMOS diode at nazc1).
  3. **Mazc3** (NMOS, d=nazc_hub, g=nazc_cap2, s=vss).
  4. **Razc0-Razc3** quad-R mesh.
  5. **Cazc0/1/2** triple-cap network (inner caps to vss).
  6. **DC bias paths** Mbp_azc_bias + Mbn_azc_hub + Mbn_azc_cap0.

R85 C cascode recipe applied
============================

  * Mcas1 / Mcas2 PMOS cascode pair on input.
  * Mcas_b + Iref_cas - pcas rail bias.
  * Mcas3_n / Mcas4_n NMOS cascode pair on mirror legs. M3/M4
    gates closed at ``d1_diode``.
  * Mcas_n_b1 + Mcas_n_b2 + Mbp_ncas_drv ncas rail.

Honest scope
============

AZC's quad-R + triple-cap mesh creates multiple internal poles +
zeros; cascoded mirror adds +15-20 dB DC gain via cascode product.
Expected gain_dc ~120-128 dB. GBW retune required as cascoded
mirror moves the dominant pole.

All seven W's BYTE-EQUAL with R86 D v124 / R85 C v123.
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


def build_nested_miller_3stage_azc_cascoded_input_mirror(
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

    # AZC-specific: triple-MOSFET archetype sizing
    W_azc: float = 4e-6,
    L_azc: float = 1e-6,

    # AZC-specific: quad-R nulling-resistor mesh
    Razc0: float = 10e3,
    Razc1: float = 10e3,
    Razc2: float = 10e3,
    Razc3: float = 10e3,

    # AZC-specific: triple-cap compensation network
    Cazc0: float = 1.5e-12,   # outer Miller-like (out <-> nazc_bias)
    Cazc1: float = 0.5e-12,   # inner cap to vss
    Cazc2: float = 0.5e-12,   # inner cap to vss

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

    # ─── R113 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # AZC cascoded-input-mirror variant. AZC's distinguishing
    # compensation network adds a quad-R mesh + triple-cap
    # backbone, exposing a clear PASSIVE_MATCHING opportunity:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Same diff-pair
    #     offset-via-Vth-mismatch argument as the cascoded-input
    #     family.
    #   * GUARD_RING on Mtail (PMOS tail current source). Same
    #     substrate-isolation argument as the rest of the family.
    #   * PASSIVE_MATCHING on the AZC quad-R nulling resistor mesh
    #     (Razc0 / Razc1 / Razc2 / Razc3). The quad-R mesh is at
    #     R85 default 10 kΩ each and forms two shared hubs
    #     (nazc_bias, nazc_hub); the AZC compensation analysis
    #     assumes the four resistors track each other, so the
    #     layout primitive MUST be a unit-R series/parallel array
    #     (Hastings Ch. 8) so the resistor mismatch scales with
    #     sqrt(unit_count) rather than per-resistor process drift.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Qu2017 AZC 3-stage opamp with BOTH a
    cascoded PMOS input pair AND a cascoded NMOS mirror load.
    AZC signature (triple-MOSFET + quad-R + triple-cap) preserved
    verbatim.
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

    # AZC delta: triple-MOSFET archetype trio
    devices.append(_mosfet("Mazc1", "nmos",
                            d="nazc1", g="d2", s="vss", b="vss",
                            W=W_azc, L=L_azc))
    devices.append(_mosfet("Mazc2_diode", "pmos",
                            d="nazc1", g="nazc1", s="vdd", b="vdd",
                            W=W_azc, L=L_azc))
    devices.append(_mosfet("Mazc3", "nmos",
                            d="nazc_hub", g="nazc_cap2", s="vss", b="vss",
                            W=W_azc, L=L_azc))

    # AZC delta: quad-R nulling-resistor mesh
    devices.append(_res("Razc0", p="nazc_bias", n="nazc_cap0",
                        value=Razc0))
    devices.append(_res("Razc1", p="nazc_bias", n="nazc_cap1",
                        value=Razc1))
    devices.append(_res("Razc2", p="nazc_hub",  n="nazc_cap1",
                        value=Razc2))
    devices.append(_res("Razc3", p="nazc_hub",  n="nazc_cap2",
                        value=Razc3))

    # AZC delta: triple-cap compensation network
    devices.append(_cap("Cazc0", p="out", n="nazc_bias", value=Cazc0))
    devices.append(_cap("Cazc1", p="nazc_cap1", n="vss", value=Cazc1))
    devices.append(_cap("Cazc2", p="nazc_cap2", n="vss", value=Cazc2))

    # DC bias paths for AZC internal nodes.
    devices.append(_mosfet("Mbp_azc_bias", "pmos",
                            d="nazc_bias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn_azc_hub", "nmos",
                            d="nazc_hub", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbn_azc_cap0", "nmos",
                            d="nazc_cap0", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R113 C: optional DRC annotations ─────────────────
    # Three annotations spanning three R111 C kinds. See the kwarg
    # docs block above for rationale.
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
                    "PMOS diff pair — Vth mismatch directly maps to "
                    "input-referred offset; common-centroid required"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) sits "
                    "adjacent to AZC's quad-R mesh and triple-cap "
                    "network; nwell guard ring tied to vdd prevents "
                    "substrate coupling into the nazc_hub / "
                    "nazc_cap1 / nazc_cap2 internal nodes"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Razc0", "Razc1", "Razc2", "Razc3"),
            extras={
                "rationale": (
                    "AZC quad-R nulling-resistor mesh — the AZC "
                    "compensation analysis assumes the four legs "
                    "track each other across the shared hubs "
                    "(nazc_bias, nazc_hub); layout primitive must "
                    "be a unit-R series/parallel array (Hastings "
                    "Ch. 8) so mismatch scales with "
                    "sqrt(unit_count)"
                ),
                "primitive": "unit_r_series_parallel_array",
            },
        ))

    return cell
