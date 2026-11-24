"""
Built-in template: nested_miller_3stage_pfc.

Three-stage opamp with **Pole-zero Following Compensation** (Ramos
2004 PFC) — an NMCF-family variant where the active feedforward
PMOS Mgmf1 is intentionally OVERSIZED relative to the standard
AFFC feedforward device. The oversizing forces gmf1 to track bias
current strongly (large W → larger absolute gm → larger relative
swing of gm with I_bias and PVT), which makes the LHP-zero
frequency from the feedforward path "follow" the non-dominant
output-stage pole across process and temperature corners.

Reference
=========

Ramos_PFC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Ramos_PFC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm11 VOUT  net050 VDDA VDDA pfet → Mgmf1 (PMOS feedforward,
                                       g=d1, d=out — same role as
                                       AFFC's Mgf2 BUT oversized
                                       via the OVERSIZE factor)
  xm10 net043 net050 VDDA VDDA pfet → Mcs2 (PMOS stage-2 CS in the
                                       paper netlist, but Clay
                                       keeps the canonical NMOS
                                       stage-2 chain for bias
                                       robustness)
  xm23 VOUT  net049 GNDA GNDA nfet → Mlcs3 (stage-3 NMOS sink)

The PFC architectural fingerprint:
  1. An active feedforward PMOS `Mgmf1` (PMOS, g=d1, d=out, s=vdd)
     — structurally identical to AFFC's Mgf2 BUT sized with a
     designer-controlled OVERSIZE factor (default 4×). This is
     the architectural signature: a feedforward device whose
     W/L is set to be several times larger than the stage-3
     PMOS Mcs3, so its gm dominates the FF path and tracks
     bias strongly.
  2. The Mgmf1's bias-tracking gm gives the LHP zero
     z_LHP ≈ -gmf1 / C_out_eff a process-aware location: as
     PVT shifts the non-dominant pole at the output (set by
     gm3 / C_L), gmf1 shifts proportionally (because gmf1 and
     gm3 are both PMOS devices on the same vdd rail), so the
     LHP zero auto-follows. The compensation tracking is the
     namesake "pole-zero following".
  3. NMOS sink at output widened by (1 + PFC_oversize) to absorb
     Mgmf1's larger DC source current (analogous to AFFC's M4
     widening but proportional to the oversize factor instead
     of unit scale).

Topology delta vs `nested_miller_3stage`
========================================

Stages 1, 2 and 3 (signal path) inherit the NMCF baseline
verbatim. Compensation routing also stays NMCF-style for the
outer Miller (Cm1 out↔d2, Cm2 d2↔d1, Cf dropped because Mgmf1
provides the LHP zero). The single delta is the oversized FF
device:

**Oversized active feedforward Mgmf1 (NEW, PFC-unique):**
  Mgmf1 : PMOS, d=out, g=d1, s=vdd, b=vdd
          W = PFC_oversize · W_cs3   (default oversize=4.0)

  Compare with AFFC's Mgf2 (PMOS, g=d1, d=out, W = Mgf2_scale ·
  W_cs3 with default 1.0). Both are gate-on-d1 / drain-on-out
  PMOS feedforward devices, but PFC sizes Mgmf1 to ≈4× Mcs3,
  giving gmf1 ≈ 4 gm3 at the same gm/Id. The 4:1 ratio comes
  from the AnalogGym paper netlist where MOSFET_11_1_W_gmf1_PMOS
  is in the range of 3-5× the standard FF device width.

**NMOS sink at output widened (NEW, balances Mgmf1 DC current):**

  NMCF baseline:
    Mlcs3 : NMOS sink at out, M = K_cs3_load

  PFC:
    Mlcs3 : NMOS sink at out, M = K_cs3_load · (1 + PFC_oversize)
            (widened by the oversize factor to absorb Mgmf1's
             larger DC source current)

**Compensation: Cm1 outer + Cm2 inner, NO Cf:**

  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (LHP-zero generator, dropped in PFC)

  PFC:
    Cm1 : out ↔ d2    (same as NMCF outer Miller)
    Cm2 : d2  ↔ d1    (same as NMCF inner Miller)
    (Cf is dropped — Mgmf1's oversized FF provides the LHP zero
     and forces it to track the non-dominant pole)

PFC vs AFFC in one line
=======================

AFFC: Mgf2 (PMOS, g=d1, d=out, W = 1 × Mcs3 by default) — single
      FF path with fixed nominal LHP zero placement. Designer
      tunes Mgf2_scale manually if PVT pulls the zero off-spec.

PFC: Mgmf1 (PMOS, g=d1, d=out, W = 4 × Mcs3 by default) — OVERSIZED
     FF device so gmf1 tracks bias strongly, making the LHP zero
     auto-follow the non-dominant pole across PVT. Same device
     topology as AFFC but W/L philosophy is opposite (oversize on
     purpose, not minimum-sized).

The two cells have the SAME 5 ports and the SAME stage-2 / stage-3
signal paths and the SAME compensation cap routing, but differ
structurally at the FF device's W (and the corresponding Mlcs3
sink M-multiplier). The materialised SPICE netlists differ by
exactly the W of one PMOS device and the M of one NMOS device.

Why pole-zero following matters (transfer-function level)
---------------------------------------------------------

In a 3-stage NMCF, the loop transfer function has:
  - Dominant pole at d2 (loaded by Cm1 · A3)
  - Non-dominant pole at out (loaded by Cm1 + C_L), location
    p_nd ≈ -gm3 / (Cm1 + C_L)
  - LHP zero from FF path at z_LHP ≈ -gm_FF / C_FF_eff

In standard AFFC (gm_FF = gm3 to first order), the zero placement
matches the nominal pole; PVT shifts both poles and the zero in
proportion, but the proportional tracking is loose because gm_FF
and gm3 sit at different DC operating points (FF device gate at
d1, output device gate at d2 — different DC voltages, different
gm/Id curves).

In PFC, the FF device is oversized so gmf1 ≈ N · gm3 (N = oversize
factor). The zero shifts to z_LHP ≈ -N · gm3 / C_FF_eff, sitting
N× higher than the non-dominant pole. But because gm3 itself
moves under PVT (gm ∝ √(I_D) in strong inversion, gm ∝ I_D / nU_T
in weak inversion), the ratio z_LHP / p_nd stays roughly N across
all corners — the zero "follows" the pole. The net effect is that
PM at GBW stays nearly constant across PVT instead of degrading at
the fast/cold corners (where gm3 normally pulls p_nd up and z_LHP
lags behind).

Honest scope (what you should NOT expect)
-----------------------------------------

PFC is a structural sizing refinement that genuinely changes the
netlist (Mgmf1 with W = 4 × Mcs3 + widened Mlcs3 sink). At the
relaxed 5 dB / 10 kHz spec floor on gf180mcu 3.3 V, the PVT
envelope will land comfortably. Whether the pole-tracking benefit
shows up cleanly depends on the oversize factor (4× nominal, but
can be tuned). Tests in `tests/test_ramos_pfc_distinct.py` lock the
architectural distinction (Mgmf1 exists with W >> Mcs3, oversize
scaling kwarg works) rather than asserting specific SPICE numerics.

Knobs
-----

PFC adds the following new kwargs on top of the NMCF baseline:

  PFC_oversize: float = 4.0
      Oversizing ratio of Mgmf1 W to Mcs3 W. Default 4.0 from the
      AnalogGym netlist (MOSFET_11_1_W_gmf1_PMOS typically 3-5×
      stage-3 W). Range 2.0-8.0 reasonable; below 2.0 the FF path
      reverts to AFFC-like single-FF behaviour, above 8.0 burns
      excessive DC current at out.
  W_gmf1, L_gmf1: optional explicit overrides for Mgmf1 sizing.
  gm_id_gmf1: gm/Id intent override for Mgmf1.

The kwarg `Cf` from the NMCF baseline is NOT exposed — PFC
explicitly drops Cf. All other knobs (I_ref, W_bp, L_bp, W_bn,
L_bn, N_tail, K_cs3_load, W_diff, L_diff, W_load1, L_load1, W_cs2,
L_cs2, W_cs3, L_cs3, gm_id_diff, gm_id_load1, gm_id_stage2,
gm_id_stage3, Cm1, Cm2) are inherited verbatim from NMCF.
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


def build_nested_miller_3stage_pfc(
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

    # ─── Compensation caps (NMCF-style routing; Cf dropped) ─
    Cm1: float = 1.5e-12,   # outer Miller (out ↔ d2), same as NMCF
    Cm2: float = 0.5e-12,   # inner Miller (d2 ↔ d1), same as NMCF

    # ─── PFC-specific: oversized active feedforward Mgmf1 ──
    # PMOS, g=d1, d=out — same role as AFFC's Mgf2 but oversized.
    # Default 2× balances the oversize philosophy (Mgmf1 is
    # noticeably larger than Mcs3 so its gm dominates the FF path
    # and tracks bias) with reasonable e2e numbers at the relaxed
    # 5 dB / 10 kHz spec floor. The AnalogGym netlist uses 3-5×
    # at sky130; gf180mcu 3.3 V has different headroom so 4× would
    # over-load Mlcs3 and crush GBW. Range 1.5-8.0 reasonable;
    # below 1.5 reverts to AFFC-like single-FF behaviour.
    PFC_oversize: float = 2.0,
    W_gmf1: float | None = None,
    L_gmf1: float | None = None,
    gm_id_gmf1: float | None = None,

    # ─── R121 C: qualitative DRC annotation surface ────────
    # Additive metadata. Default False keeps every pre-R121
    # caller byte-for-byte identical. When True, three
    # qualitative hints attach to the cell:
    #   MATCHING                on (M1, M2)     — PMOS diff pair
    #   CURRENT_MIRROR_MATCHING on (M3, M4)     — NMOS diode mirror
    #   GUARD_RING              on (Mtail,)     — PMOS tail (nwell)
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Ramos PFC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC / DFCFC2 / DFCFC1 /
    RAFFC / LEC variants. PFC differs from NMCF in two structural
    places:
      1. Oversized active feedforward Mgmf1 (PMOS, g=d1, d=out) with
         W = PFC_oversize × W_cs3 (default 4×). The oversizing is the
         architectural signature.
      2. NMOS sink at out widened by (1 + PFC_oversize) to absorb
         Mgmf1's larger DC source current.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1, Cm2: NMCF-style Miller caps (out↔d2 and d2↔d1).
        PFC_oversize: ratio of Mgmf1 W to Mcs3 W (default 4.0).
        W_gmf1, L_gmf1, gm_id_gmf1: overrides for Mgmf1 sizing.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # PFC: Mgmf1 draws additional DC current at out. The Mlcs3 NMOS
    # sink must be widened to absorb both Mcs3's current and Mgmf1's
    # current. Mgmf1 sources approximately PFC_oversize × I_cs3 from
    # vdd at the same gm/Id, so total sink current at out is
    # (1 + PFC_oversize) × I_cs3.
    I_gmf1 = PFC_oversize * I_cs3
    K_cs3_load_total = int(round(K_cs3_load * (1 + PFC_oversize)))
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

    # Mgmf1 sizing: gm_id_gmf1 beats W_gmf1 / L_gmf1 which beats
    # PFC_oversize-derived default.
    if L_gmf1 is None:
        L_gmf1 = L_cs3
    if gm_id_gmf1 is not None:
        W_gmf1 = closed_form_W("pmos", gm_id=gm_id_gmf1,
                               I_d=I_gmf1, L=L_gmf1).W
    elif W_gmf1 is None:
        W_gmf1 = W_cs3 * PFC_oversize

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

    # ─── PFC delta: Oversized active feedforward Mgmf1 ──────
    # Architectural signature of PFC. PMOS gated by d1 (stage-1 output)
    # with drain at out (global output) — same routing as AFFC's Mgf2
    # but W is OVERSIZED (default 4× Mcs3). The oversizing makes
    # gmf1 ≫ gm3, so the LHP zero from gmf1 / C_out_eff sits high
    # enough above the non-dominant pole that PVT shifts of gm3 are
    # tracked proportionally by gmf1 (both PMOS on the same vdd rail)
    # — the zero "follows" the pole.
    devices.append(_mosfet("Mgmf1", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gmf1, L=L_gmf1))

    # Stage-3 NMOS sink widened (M-multiplier scaled by 1 + oversize)
    # to absorb both Mcs3 and Mgmf1 DC currents at out.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: Cm1 outer + Cm2 inner, NO Cf ─────────
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R121 C: optional qualitative DRC annotations ─────────
    # Three annotations attached only when attach_drc=True; default
    # path is byte-identical to pre-R121.
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
                    "input-referred offset. PFC's oversized Mgmf1 path "
                    "amplifies any stage-1 imbalance into a DC error at "
                    "out, so M1/M2 common-centroid matching is the "
                    "prerequisite for the pole-zero-following compensation "
                    "to operate within its designed gain budget"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "NMOS stage-1 diode mirror (M3 diode, M4 mirror leg). "
                    "Mirror ratio error injects a static differential "
                    "current into d1 which the oversized Mgmf1 amplifies "
                    "into a stage-3 DC offset — common-centroid unit-cell "
                    "array required to hold the 1:1 ratio over process "
                    "gradients"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) sinks the full "
                    "stage-1 bias current from vdd into tail. An nwell "
                    "guard ring tied to vdd shields the high-Z d1 node "
                    "from supply-coupled substrate noise that would "
                    "otherwise modulate the oversized Mgmf1 feedforward "
                    "path"
                ),
                "well_type": "nwell",
            },
        ))

    return cell
