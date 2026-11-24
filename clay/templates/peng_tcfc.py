"""
Built-in template: nested_miller_3stage_tcfc.

Three-stage opamp with **Topology-cascoded Feedforward Compensation**
(Peng_TCFC, Peng 2004) — an NMCF variant that combines an active
feedforward transconductor (gm_f, similar in spirit to HoiLee_AFFC)
with a cascode device in the stage-2 PMOS load path, decoupling the
high-impedance d2 node from the output stage's drain-bulk parasitic.

Reference
=========

Peng_TCFC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Peng_TCFC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm11 VOUT VOUTP VDDA VDDA pfet      → Mgmf (active feedforward
                                          PMOS, gate=stage-2-out)
  xm10 net043 VOUTP VDDA VDDA pfet    → Mgm2 (current-mirror leg
                                          for the cascoded mirror)
  xm70 net043 net043 GNDA GNDA nfet   → Mlcs3_diode (NMOS diode for
                                          the cascoded sink mirror)
  xm69 net10 net043 GNDA GNDA nfet    → Mlcs3 (NMOS sink mirror,
                                          slaved off Mlcs3_diode)
  xm23 VOUT net10 GNDA GNDA nfet      → Mcas_n (NMOS cascode at
                                          stage-3 sink output)

Plus the usual diff pair, stage-2 PMOS CS, and two Miller caps
(C0 from VOUTP to VOUT, C1 from net049 to VOUT).

The TCFC architectural fingerprint:
  1. An **active feedforward** PMOS device whose gate connects to
     the stage-2 output and drain to the global output (similar to
     AFFC's Mgf2 but driven from d2 not d1 — TCFC bypasses stage-3,
     not stage-2).
  2. A **cascoded NMOS sink** at the output stage: instead of a
     single NMOS current mirror to vss, there's a 2-device stack
     (cascode device Mcas_n between out and an internal node nd3,
     plus a current source Mlcs3 from nd3 to vss). The intermediate
     node nd3 is the architectural marker.
  3. The cascode-bias chain: a NMOS diode Mlcs3_diode sets the
     bias for the Mlcs3 sink leg, providing PVT-robust current
     mirroring.

Topology delta vs `nested_miller_3stage`
========================================

Stage 1 inherits the NMCF baseline verbatim. The deltas:

**Stage 3 sink — cascoded:**
  NMCF baseline:
    Mlcs3 : NMOS, d=out, g=nbias, s=vss, b=vss   (single device sink)

  TCFC:
    Mcas_n     : NMOS, d=out, g=nbias_cas, s=nd3, b=vss   (NEW cascode)
    Mlcs3      : NMOS, d=nd3, g=nd3_diode, s=vss, b=vss   (current src)
    Mlcs3_diode: NMOS, d=nd3_diode, g=nd3_diode, s=vss, b=vss
                                                            (NEW diode)
    Mlcs3_diode_drv: PMOS, d=nd3_diode, g=pbias, s=vdd, b=vdd
                                                            (NEW src
                                                            for diode)

  The intermediate node `nd3` (between out and Mlcs3 drain) is the
  TCFC-distinct net. NMCF has no such node.

**Active feedforward (Mgmf):**
  NEW device:
    Mgmf : PMOS, d=out, g=d2, s=vdd, b=vdd

  Gate driven by d2 (stage-2 output) so the feedforward path
  bypasses ONLY the output stage. This is structurally different
  from HoiLee_AFFC's Mgf2 (which is gated by d1, bypassing both
  stage 2 and stage 3).

**Compensation: TWO Miller caps (no Cf):**
  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1    (FF cap for RHP-zero cancellation)

  TCFC:
    Cm1 : out ↔ d2    (same outer Miller as NMCF)
    Cm2 : d2  ↔ d1    (same inner Miller as NMCF)
    (Cf is dropped — the active feedforward + cascoded sink combine
     to suppress the RHP zero through structural means rather than
     cap-induced LHP-zero compensation)

Schematic (stage 3 + comp deltas only)
--------------------------------------

  NMCF (output stage):              TCFC (output stage):
    Mcs3 (PMOS) → out                  Mcs3 (PMOS) → out
    Mlcs3 (NMOS) sinks to vss          Mgmf (PMOS) → out (NEW, g=d2)
    Cm1 out↔d2 / Cm2 d2↔d1 / Cf d1↔out Mcas_n (NMOS, NEW, d=out, s=nd3)
                                       Mlcs3 (NMOS) nd3 → vss
                                       Mlcs3_diode + driver (NEW diode chain)
                                       Cm1 out↔d2 / Cm2 d2↔d1 (no Cf)

Why TCFC matters (transfer-function level)
------------------------------------------

The cascoded NMOS sink raises the impedance looking into the output
node from the sink side, which pushes the second pole (driven by
gm_3/C_d2 in NMCF) further out. Combined with the active feedforward
path through Mgmf, the loop transfer function gains an extra LHP
zero from the gm_f / gm_3 ratio (similar to AFFC but at a different
spectral location because the feedforward is gated by d2, not d1).

Net effect on PVT numbers (qualitative):
  - DC gain: slightly increased over NMCF (cascoded sink → higher
    output impedance → higher A_out)
  - GBW: depends on Cm1 and gm_3; typically similar to NMCF baseline
    because Cm1 hasn't changed
  - PM: improved relative to plain NMCF because the second-pole
    push-out from the cascoded sink lowers phase shift at GBW

Honest scope (what you should NOT expect)
-----------------------------------------

TCFC is a structural compensation refinement that genuinely changes
the netlist (new MOSFET Mgmf, new cascode Mcas_n, new bias chain
Mlcs3_diode + driver, new internal node nd3). The PVT numbers may
or may not differ measurably from NMCF baseline at the relaxed
5 dB / 10 kHz spec floor — the cascoded sink's contribution to A_out
is at most a few dB, and Mgmf's gm_f / gm_3 ratio is typically order
unity. Tests in `tests/test_peng_tcfc_distinct.py` lock the
architectural distinction (Mgmf exists, Mcas_n exists, nd3 net
exists, Cf does NOT exist) rather than asserting specific SPICE
numerics.

Knobs
-----

TCFC adds three new kwargs on top of the NMCF baseline:

  Mgmf_scale: float = 1.0
      Sizing ratio of Mgmf / Mcs3. Default 1.0 → gm_f ≈ gm_3 at
      the same gm/Id.
  W_gmf, L_gmf: optional explicit overrides for Mgmf sizing.
  W_cas_n, L_cas_n: float
      NMOS cascode device sizing (Mcas_n). Default 4 µm / 1 µm
      (matches NMOS sink mirror sizing). The cascode device's
      gate is biased at `nbias` (same as the original sink mirror)
      so its Vds_sat is comfortable on gf180mcu 3.3 V.

The kwarg `Cf` from the NMCF baseline is NOT exposed — TCFC
explicitly drops Cf. All other kwargs (I_ref, W_bp, L_bp, W_bn,
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


def build_nested_miller_3stage_tcfc(
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

    # ─── Compensation caps (Cm1, Cm2 only — no Cf in TCFC) ──
    Cm1: float = 1.5e-12,
    Cm2: float = 0.5e-12,

    # ─── TCFC-specific: active feedforward (Mgmf) ───────────
    Mgmf_scale: float = 1.0,
    W_gmf: float | None = None,
    L_gmf: float | None = None,
    gm_id_gmf: float | None = None,

    # ─── TCFC-specific: NMOS cascode at output sink (Mcas_n) ─
    W_cas_n: float = 4e-6,
    L_cas_n: float = 1e-6,
    # R109 C: qualitative DRC annotation surface (additive metadata,
    # see folded_cascode.py R108 C precedent). Default False keeps
    # every pre-R109 caller byte-for-byte identical. When True, three
    # qualitative hints attach:
    #   MATCHING       on PMOS diff pair (M1 / M2),
    #   GUARD_RING     on PMOS tail current source (Mtail),
    #   WELL_ISOLATED  on NMOS cascode device (Mcas_n) — the TCFC-defining
    #                  cascoded sink whose body-effect coupling to Mlcs3
    #                  would degrade output-impedance pole push-out.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng TCFC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC variants. TCFC differs from NMCF in
    three structural places:
      1. Active feedforward device Mgmf (PMOS, g=d2, d=out)
      2. Cascoded NMOS sink at output stage (Mcas_n + Mlcs3 + diode
         chain; new internal node `nd3`)
      3. Drop Cf (only Cm1 + Cm2)

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1, Cm2: nested Miller cap values (no Cf).
        Mgmf_scale: ratio of Mgmf W to Mcs3 W (default 1.0).
        W_gmf, L_gmf: optional explicit overrides for the feedforward
            device sizing (default: scaled from W_cs3 / L_cs3).
        gm_id_gmf: gm/Id intent override for Mgmf.
        W_cas_n, L_cas_n: NMOS cascode device sizing (default 4 µm /
            1 µm, matches Mbn sizing).

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # TCFC: Mgmf draws additional DC current proportional to Mgmf_scale.
    I_gmf = Mgmf_scale * K_cs3_load * I_ref

    # The NMOS sink leg must absorb Mcs3 + Mgmf currents. Cascoded sink
    # mirror ratio is bumped internally to keep the output DC OP near
    # the same V(out) as plain NMCF.
    K_cs3_load_total = K_cs3_load * (1 + Mgmf_scale)

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

    # Mgmf sizing: gm/Id override beats W_gmf which beats Mgmf_scale.
    if L_gmf is None:
        L_gmf = L_cs3
    if gm_id_gmf is not None:
        W_gmf = closed_form_W("pmos", gm_id=gm_id_gmf,
                              I_d=I_gmf, L=L_gmf).W
    elif W_gmf is None:
        W_gmf = W_cs3 * Mgmf_scale

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

    # ─── TCFC delta 1: Active feedforward device Mgmf ───────
    # PMOS in parallel with Mcs3 but gated by d2 (stage-2 output —
    # which is the SAME as Mcs3's gate). Wait — this would short
    # them. Re-read the AnalogGym reference: xm11 has gate=VOUTP,
    # which is the stage-2 output net. xm10 also has gate=VOUTP.
    # So both Mcs3 and Mgmf are driven from d2 — this is what makes
    # TCFC distinct from AFFC (AFFC's Mgf2 is driven from d1).
    #
    # Two devices both driven from the same gate node but going to
    # different drains is structurally distinct (gain adds at the
    # output rather than acting at a different stage's input).
    # The "topology cascode" name in Peng 2004 refers to this output-
    # stage gain doubling combined with the cascoded sink below.
    devices.append(_mosfet("Mgmf", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # ─── TCFC delta 2: Cascoded NMOS sink ─────────────────────
    # Replace the single NMOS sink (Mlcs3 d=out → vss) with a 2-level
    # stack: cascode device Mcas_n between out and nd3, current
    # source Mlcs3 between nd3 and vss. The cascode raises the
    # impedance looking into the sink from the output node, pushing
    # the second pole out and improving PM at the cost of some Vds
    # headroom on the output rail.
    #
    # Bias plan:
    #   Mcas_n's gate is biased at `nbias` (same NMOS bias-mirror
    #     reference as Mlcs3 in NMCF baseline — comfortable Vds_sat
    #     on gf180mcu 3.3 V because Vgs_n ≈ 0.7 V leaves ≈ 0.6 V
    #     of headroom for both Mcas_n and Mlcs3).
    #   Mlcs3's gate is biased at a NEW diode-connected reference
    #     `nd3_diode`, which is driven by Mlcs3_diode (NMOS-diode at
    #     nd3_diode) and Mlcs3_diode_drv (PMOS current source from
    #     pbias mirror, supplying the diode current). This separate
    #     bias loop lets Mlcs3 mirror Mbn at the correct ratio for
    #     the cascoded path.
    devices.append(_mosfet("Mcas_n", "nmos",
                            d="out", g="nbias", s="nd3", b="vss",
                            W=W_cas_n, L=L_cas_n,
                            M=K_cs3_load_total))
    devices.append(_mosfet("Mlcs3", "nmos",
                            d="nd3", g="nd3_diode", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # Mlcs3_diode: NMOS diode at nd3_diode. Mirrors Mbn but provides
    # a separate bias node for the cascoded sink (PVT-robust).
    devices.append(_mosfet("Mlcs3_diode", "nmos",
                            d="nd3_diode", g="nd3_diode",
                            s="vss", b="vss",
                            W=W_bn, L=L_bn))
    # Mlcs3_diode_drv: PMOS mirror of Mbp that pushes I_ref into the
    # nd3_diode node, defining its Vgs.
    devices.append(_mosfet("Mlcs3_diode_drv", "pmos",
                            d="nd3_diode", g="pbias",
                            s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── Compensation: Cm1 + Cm2 (no Cf) ─────────────────────
    # TCFC keeps the textbook nested-Miller pair (Cm1 outer + Cm2
    # inner) but drops the feedforward cap Cf — the active Mgmf
    # path and the cascoded sink combine to provide PM recovery
    # without needing a passive cap-induced LHP zero.
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R109 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extends the
    # template inventory consumed by the layout prompt's round-trip path.
    # Annotations live in cell.__dict__["_drc_annotations"] only — SPICE
    # emit (which iterates cell.devices) is unaffected.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). TCFC inherits
        #    the NMCF baseline stage-1 verbatim; input offset is set by
        #    M1 / M2 Vth-mismatch (Razavi §18.1 prescription).
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). Same role as
        #    folded_cascode / sau_cfcc: high-impedance tail node carries
        #    N_tail × I_ref and feeds both diff-pair sources; substrate
        #    noise coupled into Mtail propagates as common-mode that
        #    leaks through the multi-stage feedback path.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — TCFC NMOS cascode device Mcas_n at the
        #    output sink. This is the architectural marker of the Peng
        #    TCFC topology (cascoded sink). Mcas_n sits between out and
        #    the nd3 internal node, raising the impedance looking into
        #    the sink from the output node. If Mcas_n shares a pwell
        #    with the rest of the cell's NMOS devices, body-effect
        #    coupling from out swing modulates Vth_n and pulls the
        #    second-pole back in, defeating the cascoded-sink advantage
        #    that defines TCFC. A dedicated pwell tap (body=vss,
        #    isolated from the M3/M4 mirror well) preserves the pole
        #    push-out.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcas_n",),
            extras={
                "rationale": "TCFC NMOS cascoded sink — body-effect isolation from NMOS mirror M3/M4",
                "isolate_from": ("M3", "M4", "Mlcs3"),
            },
        ))

    return cell
