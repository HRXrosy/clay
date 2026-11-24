"""
Built-in template: nested_miller_3stage_nmcnr_hybrid_d1_ff.

R88 C / BL-093 hybrid d1-gated FF retrofit pilot. Validates R87 C
structural finding that cascoded mirror recipe REQUIRES a d1-gated
Mgf2-style active feedforward companion to realise its +30-35 dB
gain product.

Structural finding (R87 C closure)
==================================

After running the R85 C cascode recipe (telescopic PMOS input cascode +
cascoded NMOS mirror with DFCFC1-style diode loop at high-Z node d1)
across 10 NMCF-class variants in R85-R87 C batch rollouts, a clean
structural pattern emerged:

  * PASS variants (5/10): DFCFC2 / ACBC / LEC / PFC / DFCFC1
      ALL have an active feedforward (Mgf2 / Mgf_lec) with structure
      (PMOS, g=d1, d=out, s=vdd). gain_dc landed in 117-129 dB range.

  * FAIL variants (5/10): NMCNR / AZ / AZC / DACFC / RAFFC
      NONE have a d1-gated active FF. gain_dc ceilings at 83-87 dB.
      The cascoded mirror raises stage-1 output impedance by 4-6×, but
      WITHOUT the d1-gated FF the +30 dB cascode gain product is not
      transferred to the output node through the stage-2 → stage-3
      cascade alone.

Honest scope diagnosis: the d1-gated FF acts as a *direct gain path*
from d1 (high-Z stage-1 output) to out, bypassing stage 2's signal
inversion. Without it, the cascoded mirror's gain boost is partially
cancelled by stage 2's loading on d1, leaving gain_dc near the
pre-cascode baseline.

R88 C pilot retrofit
====================

NMCNR is the natural pilot target because:

  1. Its compensation (Rz1 nulling resistor + shared rz_n node + 2
     Miller caps to rz_n) is structurally orthogonal to the active
     FF — adding Mgf2 at out doesn't conflict with the nulling-R
     RHP-zero path.
  2. R87 C measured NMCNR cascoded baseline at gain_dc = 83.29 dB /
     GBW = 1.308 MHz (GBW already in paper +/-20% window).
  3. The 5 PASS variants picked up +30-50 dB from the d1-FF; if the
     same lift lands on NMCNR (83 dB → 115-120 dB) the structural
     finding is confirmed.

Template structure
==================

This template is a *byte-exact* extension of
`nested_miller_3stage_nmcnr_cascoded_input_mirror`:

  * Same R85 C cascode recipe (Mcas1/Mcas2 PMOS input cascode, Mcas3_n/
    Mcas4_n NMOS mirror cascode with DFCFC1-style diode loop at
    d1_diode, pcas + ncas bias rails).
  * Same NMCNR compensation backbone: Rz1 (rz_n↔out) + Cm1 (rz_n↔d2)
    + Cm2 (rz_n↔d1). NO Cf.
  * Same stage-2 NMOS CS amp + replica-bias leg.
  * Same stage-3 PMOS CS amp.

The ONE delta is the **hybrid d1-gated active feedforward**:

  * Mgf2_hybrid (PMOS, g=d1, d=out, s=vdd) — Mgf2-style FF copied
    from DFCFC2 / DFCFC1 cascoded variants.
  * Mlcs3 NMOS output sink widened by (1 + Mgf2_scale) to absorb
    the additional DC current Mgf2 sources into the output node.

The FF is "hybrid" in the sense that it grafts a *DFCFC1-class*
device onto NMCNR's *nulling-R-class* compensation backbone — the
two compensation philosophies normally exclude each other in the
literature, but they are structurally compatible at the device
level.

Architectural devices preserved verbatim
=========================================

NMCNR fingerprint that survives the d1-FF retrofit:

  1. **rz_n node** + **Rz1 resistor** (rz_n↔out) — the nulling-R
     signature.
  2. **Cm1** (rz_n↔d2) outer Miller terminating at rz_n.
  3. **Cm2** (rz_n↔d1) inner Miller ALSO terminating at rz_n.
  4. NO Cf.

R85 C cascode recipe inherited verbatim
=======================================

  * Mcas1 / Mcas2 PMOS cascode pair on input.
  * Mcas_b + Iref_cas — pcas rail bias.
  * Mcas3_n / Mcas4_n NMOS cascode pair on mirror legs. M3/M4
    gates closed at ``d1_diode`` (DFCFC1-style regulated cascode
    loop).
  * Mcas_n_b1 + Mcas_n_b2 stacked NMOS-diode pair for ncas rail.
  * Mbp_ncas_drv PMOS mirror at pbias sources I_ref.

R88 C delta
===========

  * **Mgf2_hybrid** (PMOS, g=d1, d=out, s=vdd) — the hybrid d1-FF.
  * **Mlcs3** sized by K_cs3_load * (1 + Mgf2_scale) — wider NMOS
    sink absorbs Mgf2_hybrid's DC current at out.

If gain_dc lands in [115, 120] dB AND GBW stays in [0.96, 1.44] MHz,
R87 C structural finding is CONFIRMED. R89+ can apply same retrofit
to AZ / AZC / DACFC.

Honest scope
============

Mgf2_scale = 0.10 is the R85+ tested-good value. At Mgf2_scale = 1.0
the Mgf2_hybrid current at DC cancels the main signal path (Mgf2_hybrid
gated by d1, Mcs3 gated by d2 ≈ -A2·v(d1)). At 0.1 the main A1·A2·A3
path dominates while the AC-band feedforward zero still provides the
gain lift via the d1→out direct path.
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


def build_nested_miller_3stage_nmcnr_hybrid_d1_ff(
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

    # R88 C hybrid d1-FF retrofit (DFCFC2/DFCFC1 inherited Mgf2 pattern)
    Mgf2_scale: float = 0.1,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

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
    # NMCNR + hybrid d1-FF retrofit opamp:
    #   * MATCHING on the PMOS diff pair (M1, M2). Vth mismatch maps
    #     directly to input-referred offset; the hybrid d1-FF
    #     (Mgf2_hybrid, g=d1, d=out) provides a direct gain path from
    #     d1 to out that amplifies any imbalance at M2's drain.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring
    #     tied to vdd isolates the high-Z d1 node — which now drives
    #     both Mcs2 (stage 2 input) and Mgf2_hybrid (active FF) — from
    #     supply-coupled substrate noise.
    #   * PASSIVE_MATCHING on the NMCNR Miller cap pair (Cm1, Cm2).
    #     Both share rz_n (Cm1: rz_n↔d2; Cm2: rz_n↔d1). The Rz1
    #     nulling resistor's RHP-zero migration depends on the Cm1/Cm2
    #     ratio being preserved over process; fringing-cap unit arrays
    #     make the two caps track on the same gradient.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for an NMCNR 3-stage opamp with R85 C cascode
    recipe AND a hybrid d1-gated Mgf2-style active feedforward
    (R88 C pilot retrofit validating R87 C structural finding).

    NMCNR signature (rz_n shared node + Rz1 nulling resistor + 2
    Miller caps to rz_n) preserved verbatim. R85 C cascode recipe
    preserved verbatim. NEW: Mgf2_hybrid (PMOS, g=d1, d=out) +
    Mlcs3 widened by (1 + Mgf2_scale) to absorb Mgf2 DC current.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (NMCF baseline inherited).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four signal-path devices.
        Cm1: outer Miller cap (rz_n ↔ d2). Default 4.0 pF.
        Cm2: inner Miller cap (rz_n ↔ d1). Default 0.5 pF.
        Rz1: nulling resistor (rz_n ↔ out). Default 5 kOhm.
        Mgf2_scale: R88 C hybrid d1-FF scale relative to Mcs3 stage-3
            CS amp. Default 0.1 (R85 C tested-good value; 1.0 causes
            DC cancellation, 0.1 preserves main signal path while
            adding AC-band feedforward zero).
        W_gf2, L_gf2, gm_id_gf2: Mgf2_hybrid sizing overrides.
        W_cas, L_cas, I_ref_cas: PMOS input-cascode pair sizing
            (R83 D inherited).
        W_cas_n_mirror, L_cas_n_mirror: NMOS mirror-cascode pair
            sizing (R84 D inherited).
        W_cas_n_bias, L_cas_n_bias: stacked NMOS-diode pair sizing
            for the ncas bias rail.
        I_ref_cas_n: bookkeeping placeholder; actual ncas-rail
            current set by Mbp_ncas_drv at I_ref via pbias mirror.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # R88 C: Mgf2_hybrid draws additional DC current (DFCFC1 inherited).
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(K_cs3_load * (1 + Mgf2_scale)))
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

    # Mgf2_hybrid sizing (DFCFC1 inherited pattern).
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

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

    # R88 C delta: hybrid d1-gated Mgf2-style active feedforward
    # (PMOS, g=d1, d=out, s=vdd). DFCFC1/DFCFC2 inherited pattern.
    devices.append(_mosfet("Mgf2_hybrid", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgf2_hybrid DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

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
    # MATCHING (diff pair) / GUARD_RING (Mtail) / PASSIVE_MATCHING
    # (Miller cap pair to rz_n). See kwarg docs above.
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
                    "input-referred offset. The R88 C hybrid d1-FF "
                    "Mgf2_hybrid (g=d1, d=out) provides a direct gain "
                    "path from d1 to out that amplifies any imbalance "
                    "at M2's drain into the output node"
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
                    "tied to vdd isolates the high-Z d1 node — which "
                    "now drives both Mcs2 (stage-2 input) and "
                    "Mgf2_hybrid (active FF, R88 C retrofit) — from "
                    "supply-coupled substrate noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "NMCNR Miller cap pair — both share the rz_n node "
                    "(Cm1: rz_n↔d2 outer; Cm2: rz_n↔d1 inner). The "
                    "Rz1 nulling resistor's RHP-zero migration depends "
                    "on the Cm1/Cm2 ratio being preserved over "
                    "process; fringing-cap unit arrays make the two "
                    "caps track on the same gradient"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
