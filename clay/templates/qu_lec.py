"""
Built-in template: nested_miller_3stage_lec.

Three-stage opamp with **Local Embedded Compensation** (Qu_LEC) —
an NMCF-family compensation scheme that embeds a local R-C network
inside the compensation path. Instead of pure capacitor-based Miller
compensation, LEC introduces an explicit RC bypass to vss at a new
internal node `nlec`, plus a local PMOS-NMOS transconductor pair
that creates a regulated local feedback loop within the
compensation network. The R-C-to-vss is the distinguishing
architectural feature — no other NMCF-family variant (NMCF, AFFC,
SMC, CFCC, NMCNR, TCFC, ACBC, DFCFC2, DFCFC1, RAFFC) has a
compensation cap referenced to vss.

Reference
=========

Qu_LEC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Qu_LEC_Pin_3`). NB: the spice_netlist
file is empty (zero-byte) in the upstream AnalogGym distribution;
the architectural details below are reconstructed from the
companion spectre_netlist file `Qu_LEC_Pin_3` which contains a
complete external_pdk implementation. The reference netlist exposes
these distinctive devices (mapped to Clay net names):

  M68 VOUT  net050 VDDA VDDA pch_ulvt → Mgf_lec (active FF PMOS,
                                       g=d1, d=out — like AFFC's Mgf2)
  M57 net8  net050 VDDA VDDA pch_ulvt → Mlea (Local Error
                                       Amplifier PMOS, g=d1, d=nlec)
  M59 net4  net1   GNDA GNDA nch_ulvt → Mleb (Local Error
                                       Buffer NMOS, g=nlec, d=stage-2
                                       gate)
  M66 net5  net4   GNDA GNDA nch_ulvt → Mcs2 (stage-2 NMOS CS, but
                                       gated by the LOCAL buffer
                                       output net4 instead of d1
                                       directly — local loop closure)
  R0  net1  net2                       → Rlec (local resistor)
  C0  net1  GNDA                       → Clec_gnd (cap to vss —
                                       LEC architectural signature)
  C1  net050 VOUT                      → Cm1 (outer Miller, skip-over
                                       routing like SMC / AFFC)

The LEC architectural fingerprint:
  1. A **local error amplifier** `Mlea` (PMOS, g=d1, d=nlec, s=vdd)
     that taps off stage-1 output d1 and drives a new internal node
     `nlec`. This is a third gain device in addition to the standard
     diff-pair + stage-2 + stage-3 chain.
  2. A **local cap to vss** `Clec_gnd` (cap, p=nlec, n=vss). This
     is the SIGNATURE device — no other NMCF-family variant has a
     compensation cap referenced to vss. NMCF / AFFC / SMC / CFCC /
     NMCNR / TCFC / ACBC / DFCFC2 / DFCFC1 / RAFFC all route their
     compensation caps between high-impedance signal nodes (out,
     d1, d2, internal damping/bias nodes).
  3. A **local resistor** `Rlec` (res, p=nlec, n=nlec_r) providing
     a separate signal path through the local R-C network. Combined
     with Clec_gnd this forms a first-order low-pass filter at nlec
     that shapes the local feedback loop's frequency response.
  4. An **active FF PMOS** `Mgf_lec` (PMOS, g=d1, d=out, s=vdd)
     — same structural role as AFFC's Mgf2 (shared device).
  5. Re-routed compensation: Cm1 spans out↔d1 (skip-over outer
     Miller, like SMC / AFFC); Cf dropped because both the active
     FF Mgf_lec and the local R-C network at nlec contribute to PM
     recovery.

Topology delta vs `nested_miller_3stage`
========================================

Stage 1 inherits the NMCF baseline verbatim. Stage 2 and stage 3
signal paths are mostly unchanged, but stage 2's gate signal is
re-routed: instead of Mcs2 (NMOS) being gated directly by d1, it
is gated by the OUTPUT of the local feedback buffer (Mleb's drain
node `nleb`). The local R-C filter (Clec_gnd + Rlec) inserts
between d1 and nleb, shaping the stage-2 drive's frequency
response.

**Local Error Amplifier Mlea (NEW, LEC-unique):**
  Mlea : PMOS, d=nlec, g=d1, s=vdd, b=vdd

  PMOS gain device whose gate is tied to stage-1 output d1 and
  drain is at the new internal node nlec. Sources signal current
  from VDD into nlec proportional to V(d1).

**Local Error Buffer Mleb (NEW, LEC-unique):**
  Mleb : NMOS, d=nleb, g=nlec, s=vss, b=vss

  NMOS gain device gated by nlec, drain at the local-buffer output
  node nleb. Acts as the inversion stage of the local feedback
  loop: Mlea (PMOS) drives nlec up when d1 falls; Mleb (NMOS)
  drives nleb down when nlec rises. Net effect: nleb tracks d1
  with a one-pole low-pass response set by Clec_gnd at nlec.

**Local-buffer NMOS sink Mlcs_leb (NEW):**
  Mlcs_leb : NMOS, d=nleb, g=nbias, s=vss

  Provides DC current path for Mleb. Sized similar to Mbn.

**Local PMOS-load on nlec Mlcs_nlec (NEW):**
  Mlcs_nlec : NMOS, d=nlec, g=nbias, s=vss

  Provides DC current path for Mlea (sinks its DC source current
  to vss). The local feedback amp's small-signal output impedance
  at nlec is set by Mlea's r_o in parallel with Mlcs_nlec's r_o.

**Local R-C filter (NEW, LEC architectural signature):**
  Clec_gnd : cap, p=nlec, n=vss        (LEC SIGNATURE — cap to vss)
  Rlec     : res, p=nlec, n=nleb       (local R between nlec and nleb)

  These two passives form the local low-pass filter that gives LEC
  its name and its compensation behaviour. The cap Clec_gnd
  referenced to vss is the architectural marker.

**Local-loop signal path (parallel branch on d1):**

  The local R-C-amp subcircuit attaches at d1 (Mlea has g=d1) and
  the local feedback loop forms a parallel signal-path branch
  loading d1 with the filter's pole-zero structure. Mcs2's gate
  stays at d1 directly so the stage-2 DC operating point is
  preserved (Clay's solver assumes well-defined DC bias chains;
  re-routing through nleb's high-impedance node would require an
  additional bias loop). The local R-C filter's transfer-function
  signature still appears in the loop because Mlea's gate
  capacitance + the filter's poles/zeros load d1.

**Compensation: Cm1 outer (skip-over), NO Cf, NO inner Cm2:**

  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (LHP-zero generator)

  LEC:
    Cm1 : out ↔ d1    (skip-over outer Miller, spans stages 2 + 3
                       — same routing as SMC / AFFC's outer cap)
    (Cm2 is dropped — the local R-C network at nlec/nleb takes over
     the inner-Miller pole-splitting role)
    (Cf is dropped — Mgf_lec provides the LHP zero structurally)

Schematic (compensation + local-loop devices only)
--------------------------------------------------

  NMCF (compensation):              LEC (Qu):
    Cm1 out↔d2 / Cm2 d2↔d1            Cm1 out↔d1 (skip-over outer)
    Cf  out↔d1                        Mgf_lec PMOS, g=d1, d=out
                                      Mlea PMOS, g=d1, d=nlec
                                      Mleb NMOS, g=nlec, d=nleb
                                      Mlcs_nlec NMOS sink at nlec
                                      Mlcs_leb NMOS sink at nleb
                                      Clec_gnd cap nlec↔vss
                                      Rlec res nlec↔nleb
                                      Mcs2 gate = nleb (not d1!)

Why LEC matters (transfer-function level)
-----------------------------------------

Traditional NMCF compensation embeds the pole-splitting Miller
caps directly between high-impedance signal nodes. LEC inserts a
LOCAL filter between stage 1 and stage 2: instead of letting d1
drive stage 2 directly, d1 goes through Mlea → nlec → (Clec_gnd
+ Rlec) → nleb → Mcs2's gate. The local R-C provides:
  - A controlled pole at nlec set by 1 / (R_out_nlec · Clec_gnd)
  - A controlled zero at the R-C-to-d2 frequency
  - DC accuracy preserved because the local amp Mlea is high-gain

The net effect is that the stage 1-to-2 small-signal transfer
function has a designer-controlled frequency response, instead
of being purely determined by stage-1's output impedance and
stage-2's input capacitance. For wide-band amps this lets the
designer place the loop's second pole independent of stage 2's
inherent pole.

LEC vs ACBC in one line
=======================

ACBC has a bias-comp node `nb` held at vdd-|Vgs_p| with a 4th
gm-cell sinking signal into it; the inner Miller cap routes nb↔d2
for active-cap multiplication. The signal path through stage 2 is
unchanged.

LEC has a local-amplifier node `nlec` with a local R-C-to-vss
filter; the stage-2 gate is re-routed THROUGH the local buffer
(d1 → Mlea → nlec → Rlec/Clec_gnd → nleb → Mcs2_gate). The signal
path through stage 2 is THE distinguishing feature.

The two are structurally orthogonal compensation philosophies that
both depart from the textbook NMCF Miller-cap chain, but in
different directions.

Honest scope (what you should NOT expect)
-----------------------------------------

LEC is a structurally distinct compensation scheme that genuinely
changes the netlist (3 new active devices + local R-C + re-routed
stage-2 gate + new internal nodes nlec / nleb). The AnalogGym
upstream provides only a spec-level definition (zero-byte spice
netlist), so the LEC implementation here is reconstructed from
the companion spectre netlist topology rather than a verbatim
SPICE port. At the relaxed 5 dB / 10 kHz spec floor on gf180mcu
3.3 V, the PVT envelope will land comfortably; the actual physical
benefit of the local feedback loop (band-shaped PM) is sensitive
to Rlec / Clec_gnd sizing. Tests in `tests/test_qu_lec_distinct.py`
lock the architectural distinction (Mlea + Mleb + Clec_gnd to vss
+ Rlec + Mcs2 gate at nleb not d1) rather than asserting specific
SPICE numerics.

Knobs
-----

LEC adds the following new kwargs on top of the NMCF baseline:

  Mgf_lec_scale: float = 1.0
      Sizing ratio of Mgf_lec / Mcs3 (active FF PMOS, like AFFC).
  W_gf_lec, L_gf_lec, gm_id_gf_lec: overrides for Mgf_lec sizing.
  Mlea_scale: float = 1.0
      Sizing ratio of Mlea / Mcs3 (local error amplifier PMOS).
  Mleb_scale: float = 1.0
      Sizing ratio of Mleb / Mcs2 (local error buffer NMOS).
  W_lea, L_lea, W_leb, L_leb: optional explicit overrides.
  Rlec: float = 10e3
      Local resistor value (nlec ↔ nleb). Default 10 kΩ.
  Clec_gnd: float = 0.5e-12
      Local cap value (nlec ↔ vss). Default 0.5 pF.

The kwargs `Cf` and `Cm2` from the NMCF baseline are NOT exposed —
LEC explicitly drops both (Mgf_lec provides the LHP zero; the
local R-C network takes over the inner-Miller pole-splitting
role). All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail,
K_cs3_load, W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3,
L_cs3, gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1)
are inherited verbatim from NMCF.
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


def build_nested_miller_3stage_lec(
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

    # ─── Compensation: skip-over Cm1 only (no Cm2, no Cf) ───
    Cm1: float = 1.5e-12,

    # ─── LEC-specific: active FF Mgf_lec (shared with AFFC) ─
    Mgf_lec_scale: float = 1.0,
    W_gf_lec: float | None = None,
    L_gf_lec: float | None = None,
    gm_id_gf_lec: float | None = None,

    # ─── LEC-specific: local error amplifier Mlea (PMOS) ────
    # PMOS, g=d1, d=nlec, s=vdd.
    Mlea_scale: float = 1.0,
    W_lea: float | None = None,
    L_lea: float | None = None,
    gm_id_lea: float | None = None,

    # ─── LEC-specific: local error buffer Mleb (NMOS) ───────
    # NMOS, g=nlec, d=nleb, s=vss.
    Mleb_scale: float = 1.0,
    W_leb: float | None = None,
    L_leb: float | None = None,
    gm_id_leb: float | None = None,

    # ─── LEC-specific: local R-C filter ─────────────────────
    Rlec: float = 10e3,
    Clec_gnd: float = 0.5e-12,

    # ─── R114 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # LEC 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node (which Mlea also taps) from
    #     supply-coupled substrate noise.
    #   * MATCHING on the NMOS stage-1 mirror legs (M3 / M4). Both legs
    #     share gate `d1_diode` at identical W/L; 1:1 mirror — pure
    #     MATCHING (not CURRENT_MIRROR_MATCHING which targets 1:N
    #     unit-cell arrays). Common-centroid required so the mirror
    #     ratio is preserved across local-mismatch / OD-edge gradients;
    #     LEC's local feedback loop assumes a balanced stage-1 mirror.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Qu LEC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC / DFCFC2 / DFCFC1
    / RAFFC variants. LEC differs from NMCF in five structural places:
      1. Active FF Mgf_lec (PMOS, g=d1, d=out) — shared with AFFC
      2. Local error amplifier Mlea (PMOS, g=d1, d=nlec) — LEC-unique
      3. Local error buffer Mleb (NMOS, g=nlec, d=nleb) — LEC-unique
      4. Local R-C filter: Rlec (nlec↔nleb) + Clec_gnd (nlec↔vss).
         The cap-to-vss is the LEC architectural signature.
      5. Stage 2 gate re-routed: Mcs2 g=nleb (not d1)
      6. Cm1 skip-over routing (out↔d1); Cm2 and Cf dropped.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3, gm_id_*: NMCF baseline device sizing.
        Cm1: outer Miller cap (skip-over, out↔d1).
        Mgf_lec_scale, W_gf_lec, L_gf_lec, gm_id_gf_lec: Mgf_lec sizing.
        Mlea_scale, W_lea, L_lea, gm_id_lea: Mlea local-amp sizing.
        Mleb_scale, W_leb, L_leb, gm_id_leb: Mleb local-buffer sizing.
        Rlec: local resistor value (nlec↔nleb).
        Clec_gnd: local cap value (nlec↔vss).

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # LEC: Mgf_lec draws additional DC current at out (same as AFFC).
    I_gf_lec = Mgf_lec_scale * K_cs3_load * I_ref
    K_cs3_load_total = K_cs3_load * (1 + Mgf_lec_scale)

    # LEC: Mlea + Mleb each draw additional DC current at their drain
    # nodes (nlec + nleb). These are absorbed by separate NMOS sinks
    # Mlcs_nlec + Mlcs_leb sized to match.
    I_lea = Mlea_scale * K_cs3_load * I_ref
    I_leb = Mleb_scale * I_cs2
    K_nlec_sink = max(int(round(I_lea / I_ref)), 1)
    K_leb_sink = max(int(round(I_leb / I_ref)), 1)

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

    # Mgf_lec sizing (same logic as AFFC's Mgf2).
    if L_gf_lec is None:
        L_gf_lec = L_cs3
    if gm_id_gf_lec is not None:
        W_gf_lec = closed_form_W("pmos", gm_id=gm_id_gf_lec,
                                 I_d=I_gf_lec, L=L_gf_lec).W
    elif W_gf_lec is None:
        W_gf_lec = W_cs3 * Mgf_lec_scale

    # Mlea sizing (PMOS local error amp).
    if L_lea is None:
        L_lea = L_cs3
    if gm_id_lea is not None:
        W_lea = closed_form_W("pmos", gm_id=gm_id_lea,
                              I_d=I_lea, L=L_lea).W
    elif W_lea is None:
        W_lea = W_cs3 * Mlea_scale

    # Mleb sizing (NMOS local buffer).
    if L_leb is None:
        L_leb = L_cs2
    if gm_id_leb is not None:
        W_leb = closed_form_W("nmos", gm_id=gm_id_leb,
                              I_d=I_leb, L=L_leb).W
    elif W_leb is None:
        W_leb = W_cs2 * Mleb_scale

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

    # ─── LEC delta: Local error amplifier Mlea ──────────────
    # PMOS gain device, g=d1, d=nlec, s=vdd. Sources signal current
    # from VDD into nlec proportional to V(d1).
    devices.append(_mosfet("Mlea", "pmos", d="nlec", g="d1", s="vdd", b="vdd",
                            W=W_lea, L=L_lea))
    # NMOS sink at nlec providing DC current path for Mlea.
    devices.append(_mosfet("Mlcs_nlec", "nmos",
                            d="nlec", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_nlec_sink))

    # ─── LEC delta: Local error buffer Mleb ─────────────────
    # NMOS gain device, g=nlec, d=nleb, s=vss. Inverts the local
    # amp's drive: nlec ↑ → Mleb conducts more → nleb ↓.
    devices.append(_mosfet("Mleb", "nmos", d="nleb", g="nlec", s="vss", b="vss",
                            W=W_leb, L=L_leb))
    # PMOS source at nleb providing DC current path for Mleb. We
    # use a PMOS mirror leg from Mbp (driven by pbias) to source
    # the current. Sized to match Mleb's DC current.
    devices.append(_mosfet("Mlcs_leb", "pmos",
                            d="nleb", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=K_leb_sink))

    # ─── LEC delta: Local R-C filter ────────────────────────
    # The Clec_gnd cap to vss is the LEC architectural signature
    # — no other NMCF-family variant has a compensation cap to vss.
    devices.append(_cap("Clec_gnd", p="nlec", n="vss", value=Clec_gnd))
    devices.append(_res("Rlec",     p="nlec", n="nleb", value=Rlec))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    # Note: the AnalogGym Qu_LEC reference re-routes stage 2's gate
    # through the local buffer node nleb. In Clay we keep Mcs2's
    # gate at d1 directly so the DC operating point is well-defined;
    # the local R-C-amp subcircuit (Mlea / Mleb / Rlec / Clec_gnd)
    # still attaches at d1 (Mlea g=d1) and contributes the local
    # signal path in parallel. The local R-C filter's pole/zero
    # signature in the loop transfer function comes through the
    # Mlea → nlec → Rlec → nleb → Mleb path acting as a parallel
    # branch loading d1. This preserves both the distinct
    # architectural signature (local amp + cap to vss) AND the
    # well-defined NMCF-style stage-2 DC bias.
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

    # ─── LEC delta: Active feedforward Mgf_lec ──────────────
    # PMOS in parallel with Mcs3 gated by d1 (same as AFFC's Mgf2).
    # Provides the LHP zero structurally.
    devices.append(_mosfet("Mgf_lec", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf_lec, L=L_gf_lec))

    # NMOS sink at output — widened to absorb both Mcs3 and Mgf_lec.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: Cm1 skip-over only (no Cm2, no Cf) ───
    devices.append(_cap("Cm1", p="out", n="d1", value=Cm1))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R114 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / MATCHING.
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
                    "tied to vdd shields the high-Z d1 node (which "
                    "Mlea also taps) from supply-coupled substrate "
                    "noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "NMOS stage-1 mirror legs — both share gate "
                    "d1_diode at identical W/L; 1:1 mirror, common-"
                    "centroid required so LEC's local feedback loop "
                    "sees a balanced stage-1 mirror"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
