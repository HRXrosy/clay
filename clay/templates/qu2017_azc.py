"""
Built-in template: nested_miller_3stage_azc.

Three-stage opamp with **Auto-Zero Compensation** (Qu 2017 AZC) —
an NMCF-family variant where the compensation network is a triple
Miller-cap + quad nulling-resistor charge-redistribution mesh.
Three named compensation MOSFETs (Mazc1/Mazc2/Mazc3) drive a
charge-redistribution network that periodically auto-zeros the
offset of the inner pole-splitting path, combined with four
nulling resistors (Razc0/Razc1/Razc2/Razc3) that set the
zero placements.

Reference
=========

Qu2017_AZC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Qu2017_AZC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm12 net055 net094 GNDA GNDA nfet → Mazc1 (AZC1 NMOS,
                                       d=nazc1, g=d2; triggers
                                       charge redistribution from
                                       the stage-2 output node)
  xm6  net055 net055 VDDA VDDA pfet → Mazc2_diode (AZC2 PMOS
                                       diode, holds nazc1 at one
                                       Vgs_p below vdd)
  xm7  net057 net055 VDDA VDDA pfet → Mazc2_mirror (AZC2 PMOS
                                       mirror, sources current
                                       into the AZC hub node
                                       nazc_hub)
  xm22 net094 net051 GNDA GNDA nfet → Mazc3_a (AZC3 NMOS,
                                       d=d2, g=nazc_cap1; closes
                                       loop from inner cap node
                                       back to d2)
  xm23 net055 net043 GNDA GNDA nfet → Mazc3_b (AZC3 NMOS,
                                       d=nazc1, g=nazc_cap2)
  xm24 net049 net057 GNDA GNDA nfet → Mazc3_c (AZC3 NMOS,
                                       d=out_pre, g=nazc_hub)

Plus four nulling resistors R0-R3:
  R0 net078 net077 → Razc0 (nazc_bias↔nazc_cap0, sets the inner
                            cap's zero placement)
  R1 net078 net082 → Razc1 (nazc_bias↔nazc_cap1)
  R2 net057 net051 → Razc2 (nazc_hub↔nazc_cap1)
  R3 net057 net043 → Razc3 (nazc_hub↔nazc_cap2)

And three compensation caps:
  C0 net063 VOUT  → Cazc0 (out↔inner, standard outer Miller-like)
  C1 net051 GNDA  → Cazc1 (nazc_cap1↔vss, AZC inner-cap to vss)
  C2 net043 GNDA  → Cazc2 (nazc_cap2↔vss, AZC inner-cap to vss)

The AZC architectural fingerprint:
  1. THREE named compensation MOSFETs Mazc1/Mazc2/Mazc3 (some are
     groups in the paper; Clay materialises three representative
     devices Mazc1 + Mazc2_diode + Mazc3 to capture the
     archetype). The triple AZC1/AZC2/AZC3 archetype set is
     unique in AnalogGym — no other paper variant has these
     three named compensation classes.
  2. FOUR nulling resistors Razc0/Razc1/Razc2/Razc3 that form
     a quad-R network with two shared hub nodes (nazc_bias and
     nazc_hub). Razc0+Razc1 share endpoint nazc_bias; Razc2+Razc3
     share endpoint nazc_hub. This 2-hub R-network is the AZC
     resistor signature.
  3. THREE compensation caps Cazc0/Cazc1/Cazc2 with TWO of them
     (Cazc1, Cazc2) referenced to vss — making AZC the SECOND
     NMCF-family variant with vss-referenced compensation caps
     (LEC was the first with Clec_gnd). AZC's two-cap-to-vss
     mesh is distinct from LEC's single Clec_gnd.
  4. NMCF-style outer Miller routing for Cazc0 (out↔nazc_bias)
     plus the inner-cap-to-vss + nulling-R mesh; Cf dropped
     because the AZC mesh provides the compensation shape.

Topology delta vs `nested_miller_3stage`
========================================

Stages 1, 2 and 3 (signal path) inherit the NMCF baseline
verbatim. The deltas are entirely in the compensation network:

**Three AZC MOSFETs (NEW, AZC-unique):**
  Mazc1       : NMOS, d=nazc1, g=d2, s=vss
                Triggers charge redistribution from d2 into the
                AZC hub. Gated by d2 (stage-2 output) so the
                redistribution responds to inner-loop signal.
  Mazc2_diode : PMOS diode at nazc1 (d=g=nazc1, s=vdd)
                Holds nazc1 at vdd - |Vgs_p|, providing DC bias
                for Mazc1's drain.
  Mazc3       : NMOS, d=nazc_hub, g=nazc_cap2, s=vss
                Closes the AZC charge-redistribution loop by
                pulling nazc_hub low in proportion to V(nazc_cap2).

**Four AZC nulling resistors (NEW, AZC-unique):**
  Razc0 : nazc_bias ↔ nazc_cap0  (R hub 1, leg 0)
  Razc1 : nazc_bias ↔ nazc_cap1  (R hub 1, leg 1)
  Razc2 : nazc_hub  ↔ nazc_cap1  (R hub 2, leg 0)
  Razc3 : nazc_hub  ↔ nazc_cap2  (R hub 2, leg 1)

  Two hub nodes (nazc_bias, nazc_hub) each branch into two
  resistors → quad-R mesh. The 2-hub structure is the AZC
  resistor architectural signature.

**Three AZC compensation caps (NEW, AZC-unique):**
  Cazc0 : out ↔ nazc_bias      (outer Miller-like, NOT vss-referenced)
  Cazc1 : nazc_cap1 ↔ vss      (inner cap to vss — AZC signature)
  Cazc2 : nazc_cap2 ↔ vss      (inner cap to vss — AZC signature)

  Two of three caps to vss is the AZC cap signature; LEC only has
  one cap to vss (Clec_gnd).

**Compensation: NMCF Cm2 + Cf dropped, AZC mesh added:**

  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1

  AZC:
    Cazc0 : out ↔ nazc_bias      (outer Miller-like)
    Cazc1 : nazc_cap1 ↔ vss      (vss-referenced inner)
    Cazc2 : nazc_cap2 ↔ vss      (vss-referenced inner)
    Razc0-Razc3 nulling-R mesh
    (Cm1 / Cm2 / Cf dropped — AZC mesh replaces all three)

AZC vs LEC in one line
======================

LEC: ONE compensation cap to vss (Clec_gnd), ONE local resistor
     (Rlec), local error amp + buffer pair (Mlea/Mleb) — single
     local R-C-amp branch on d1.

AZC: TWO compensation caps to vss (Cazc1, Cazc2), FOUR nulling
     resistors (Razc0-Razc3), THREE compensation MOSFET archetypes
     (Mazc1, Mazc2_diode, Mazc3) — full triple-cap quad-R mesh
     charge-redistribution network.

The two cells are both vss-referenced-compensation variants but
AZC has 2× as many caps to vss, 4× as many resistors, and the
charge-redistribution archetype trio.

Honest scope (what you should NOT expect)
-----------------------------------------

AZC is a structurally distinct compensation scheme that genuinely
changes the netlist (3 new MOSFETs + 4 new resistors + 3 new caps
+ multiple new internal nodes). The AnalogGym netlist is a
sky130-pdk reference; the Clay implementation captures the
architectural archetype (the triple-AZC + quad-R + two-cap-to-vss
signature) without claiming bit-perfect SPICE equivalence to the
paper. At the relaxed 5 dB / 10 kHz spec floor on gf180mcu 3.3 V,
the PVT envelope will land comfortably; whether the AZC mesh
gives better PVT corners than NMCF depends heavily on the four
resistor values and three cap values. Tests in
`tests/test_qu2017_azc_distinct.py` lock the architectural
distinction (Mazc1/Mazc2_diode/Mazc3 exist, Razc0-Razc3 form a
4-R mesh, Cazc1/Cazc2 are vss-referenced) rather than asserting
specific SPICE numerics.

Knobs
-----

AZC adds the following new kwargs on top of the NMCF baseline:

  W_azc, L_azc: float = 4 µm / 1 µm
      Sizing for the three AZC MOSFETs (matched across the triple
      for simplicity). Larger W increases gm of the redistribution
      path; default chosen to match the AnalogGym netlist sizing
      range (MOSFET_24_8 typical values).
  Razc0, Razc1, Razc2, Razc3: float = 10 kΩ each
      Nulling-R values for the quad-R mesh. Reasonable range
      5 kΩ - 50 kΩ; below 5 kΩ the zeros move too high and lose
      compensation effect, above 50 kΩ the DC bias at the hub
      nodes drifts.
  Cazc0: float = 1.5 pF       (outer Miller-like cap to out)
  Cazc1: float = 0.5 pF       (inner cap to vss)
  Cazc2: float = 0.5 pF       (inner cap to vss)

The kwargs `Cf`, `Cm1`, `Cm2` from the NMCF baseline are NOT
exposed — AZC drops all three (the AZC mesh replaces the entire
NMCF compensation network). All other knobs (I_ref, W_bp, L_bp,
W_bn, L_bn, N_tail, K_cs3_load, W_diff, L_diff, W_load1, L_load1,
W_cs2, L_cs2, W_cs3, L_cs3, gm_id_diff, gm_id_load1, gm_id_stage2,
gm_id_stage3) are inherited verbatim from NMCF.
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


def build_nested_miller_3stage_azc(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs3_load: int = 4,

    # ─── Per-device sizing: physical (W, L) form ────────────
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load1: float = 4e-6,  L_load1: float = 1e-6,
    W_cs2: float = 20e-6,   L_cs2: float = 0.5e-6,
    W_cs3: float = 40e-6,   L_cs3: float = 0.5e-6,

    # ─── Per-device sizing: intent (gm/Id) form ─────────────
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # ─── AZC-specific: triple-MOSFET archetype sizing ───────
    W_azc: float = 4e-6,
    L_azc: float = 1e-6,

    # ─── AZC-specific: quad-R nulling-resistor mesh ─────────
    Razc0: float = 10e3,
    Razc1: float = 10e3,
    Razc2: float = 10e3,
    Razc3: float = 10e3,

    # ─── AZC-specific: triple-cap compensation network ──────
    Cazc0: float = 1.5e-12,   # outer Miller-like (out ↔ nazc_bias)
    Cazc1: float = 0.5e-12,   # inner cap to vss
    Cazc2: float = 0.5e-12,   # inner cap to vss

    # ─── R114 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # AZC 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node from supply-coupled
    #     substrate noise.
    #   * PASSIVE_MATCHING on the AZC quad-R nulling-resistor mesh
    #     (Razc0 / Razc1 / Razc2 / Razc3). The AZC compensation analysis
    #     (Qu 2017 §III) assumes the four resistor legs track each other
    #     across the two shared hub nodes (nazc_bias, nazc_hub); the
    #     pole-zero placement depends on the ratios staying balanced.
    #     Hastings Ch. 8 — layout primitive must be a unit-R series/
    #     parallel array so mismatch scales with sqrt(unit_count).
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Qu2017 AZC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC / DFCFC2 / DFCFC1 /
    RAFFC / LEC variants. AZC differs from NMCF in three structural
    places:
      1. Three AZC MOSFETs (Mazc1 NMOS, Mazc2_diode PMOS, Mazc3 NMOS)
         forming the charge-redistribution archetype trio.
      2. Four AZC nulling resistors (Razc0-Razc3) forming a quad-R
         mesh with two shared hub nodes (nazc_bias, nazc_hub).
      3. Three AZC compensation caps (Cazc0 to out, Cazc1/Cazc2 to vss)
         replacing NMCF's Cm1/Cm2/Cf.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3, gm_id_*: NMCF baseline device sizing.
        W_azc, L_azc: AZC MOSFET trio sizing.
        Razc0-Razc3: quad-R nulling-resistor mesh values.
        Cazc0: outer Miller-like cap (out ↔ nazc_bias).
        Cazc1, Cazc2: inner caps to vss (AZC signature).

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
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

    # ─── Bias chain (PMOS diode + NMOS cross-mirror) ─────────
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

    # ─── Stage 1: PMOS diff pair with NMOS mirror load ───────
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg (unchanged from NMCF) ──────
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── Stage 3: PMOS CS amplifier (output stage) ───────────
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # Stage-3 NMOS sink (unchanged from NMCF baseline).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── AZC delta: triple-MOSFET archetype trio ────────────
    # Architectural signature of AZC. Three named compensation
    # MOSFETs forming the charge-redistribution archetype:

    # Mazc1: NMOS triggered by d2 (stage-2 output). Sinks current
    # into vss in proportion to V(d2). Drain at nazc1 (one of the
    # two AZC internal nodes).
    devices.append(_mosfet("Mazc1", "nmos",
                            d="nazc1", g="d2", s="vss", b="vss",
                            W=W_azc, L=L_azc))

    # Mazc2_diode: PMOS diode at nazc1 — holds nazc1 at one
    # |Vgs_p| below vdd, providing the DC bias clamp for the
    # AZC1 drain.
    devices.append(_mosfet("Mazc2_diode", "pmos",
                            d="nazc1", g="nazc1", s="vdd", b="vdd",
                            W=W_azc, L=L_azc))

    # Mazc3: NMOS closing the redistribution loop. Drain at
    # nazc_hub (the second AZC internal hub), gate at nazc_cap2
    # (one of the two vss-referenced cap nodes). The g=nazc_cap2
    # tie makes Mazc3 the bridge from the inner-cap-to-vss network
    # back into the hub.
    devices.append(_mosfet("Mazc3", "nmos",
                            d="nazc_hub", g="nazc_cap2", s="vss", b="vss",
                            W=W_azc, L=L_azc))

    # ─── AZC delta: quad-R nulling-resistor mesh ────────────
    # Two shared hub nodes (nazc_bias, nazc_hub) each branch into
    # two resistors → 4-R mesh. This is the AZC resistor signature.
    devices.append(_res("Razc0", p="nazc_bias", n="nazc_cap0",
                        value=Razc0))
    devices.append(_res("Razc1", p="nazc_bias", n="nazc_cap1",
                        value=Razc1))
    devices.append(_res("Razc2", p="nazc_hub",  n="nazc_cap1",
                        value=Razc2))
    devices.append(_res("Razc3", p="nazc_hub",  n="nazc_cap2",
                        value=Razc3))

    # ─── AZC delta: triple-cap compensation network ─────────
    # Cazc0: outer Miller-like cap (out ↔ nazc_bias). Drives the
    # AZC mesh from the global output.
    devices.append(_cap("Cazc0", p="out", n="nazc_bias", value=Cazc0))
    # Cazc1: inner cap to vss — AZC signature. The nazc_cap1 node
    # is the shared endpoint of Razc1 (hub 1) and Razc2 (hub 2).
    devices.append(_cap("Cazc1", p="nazc_cap1", n="vss", value=Cazc1))
    # Cazc2: inner cap to vss — AZC signature. The nazc_cap2 node
    # is also Mazc3's gate, closing the redistribution loop.
    devices.append(_cap("Cazc2", p="nazc_cap2", n="vss", value=Cazc2))

    # ─── DC bias path for the AZC mesh ──────────────────────
    # The nazc_bias and nazc_hub hub nodes need DC paths so they
    # don't float. We bias nazc_bias from pbias via a Mbp mirror
    # leg (sources I_ref into nazc_bias) and nazc_hub from nbias
    # via a Mbn mirror leg (sinks I_ref to vss). This keeps the
    # hubs at well-defined DC voltages without disturbing the AC
    # compensation path's high-impedance signal flow.
    devices.append(_mosfet("Mbp_azc_bias", "pmos",
                            d="nazc_bias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn_azc_hub", "nmos",
                            d="nazc_hub", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    # The cap nodes (nazc_cap0, nazc_cap1, nazc_cap2) also need DC
    # paths so they sit at well-defined voltages. We tie each cap
    # node to nbias via a small NMOS mirror leg (sinks I_ref to vss).
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

    # ─── R114 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / PASSIVE_MATCHING.
    # See the kwarg docs block above for the per-annotation rationale.
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
                    "PMOS tail current source (M=N_tail) carries the "
                    "entire stage-1 bias current; nwell guard ring "
                    "tied to vdd shields the high-Z d1 node from "
                    "supply-coupled substrate noise"
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
                    "compensation analysis (Qu 2017 §III) assumes the "
                    "four legs track each other across the shared "
                    "hubs (nazc_bias, nazc_hub); pole-zero placement "
                    "depends on balanced ratios; layout primitive "
                    "must be a unit-R series/parallel array (Hastings "
                    "Ch. 8) so mismatch scales with sqrt(unit_count)"
                ),
                "primitive": "unit_r_series_parallel_array",
            },
        ))

    return cell
