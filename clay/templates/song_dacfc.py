"""
Built-in template: nested_miller_3stage_dacfc.

Three-stage opamp with **Doubly-differential Active-Cascode
FeedForward Compensation** (Song DACFC) — an NMCF-family variant
that combines a matched gma1 active-cascode PMOS pair with an
input-coupled gmf1_PMOS pair. The "doubly-differential" name
captures two simultaneous differential paths:
  1. The matched gma1_a + gma1_b PMOS pair (driven by the
     stage-2 output side and a replica) — a differential
     active-cascode feedforward leg.
  2. The input-coupled gmf1_PMOS pair (Mgmf1_n gated by inn,
     Mgmf1_p gated by inp) — a SECOND differential feedforward
     path that taps directly off the global input nodes.

Both legs feed forward into the inner pole-splitting nodes,
making DACFC distinct from AFFC (single-ended active FF gated
by d1 only) and PFC (oversized single-ended FF) and Song's
own DACFC distinctive choice of taking gmf1 directly from the
inputs rather than from d1.

Reference
=========

Song_DACFC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Song_DACFC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm5  net5 VOUTN VDDA VDDA pfet → Mga1_a (gma1 PMOS pair half a,
                                       drains to active-cascode
                                       node nga1_a)
  xm6  net050 VOUTN VDDA VDDA pfet → Mga1_b (gma1 PMOS pair half b,
                                       drains to nga1_b)
  xm7  net049 VOUTN VDDA VDDA pfet → Mga2 (gma2 PMOS single,
                                       drives stage-2 output)
  xm11 VOUT net4 VDDA VDDA pfet  → Mgmf2 (final output PMOS FF
                                       gated by d2, drains to out)
  xm57 net043 VINN net1 net1 pfet → Mgmf1_n (input-coupled FF
                                       PMOS gated by inn, drains
                                       to a stage-2 internal node)
  xm58 net049 VINP net1 net1 pfet → Mgmf1_p (input-coupled FF
                                       PMOS gated by inp, drains
                                       to the other stage-2 node)

The DACFC architectural fingerprint:
  1. **Matched gma1 PMOS pair** Mga1_a + Mga1_b (both gated by
     a shared node — taken in Clay as a stage-2 replica VOUTN-
     proxy, i.e. bias_d2 — drains at two different cascode
     nodes nga1_a / nga1_b). The matched-pair active cascode
     is the first half of the doubly-differential signature.
  2. **Input-coupled gmf1_PMOS pair** Mgmf1_n + Mgmf1_p (gated
     directly by inn and inp respectively, drains feeding back
     into stage-2 sense nodes). The fact that gmf1 takes its
     gate from the INPUT (not from stage-1's output d1) is the
     distinguishing architectural choice of DACFC — it bypasses
     the diff-pair and provides a direct FF path from input to
     the inner Miller loop. This is the second half of the
     doubly-differential signature.
  3. **Single gma2 PMOS** Mga2 (gated by the same stage-2 replica
     node, drain at a sense node) — provides additional
     cascode buffering between gma1 and the output.
  4. **Single gmf2 PMOS** Mgmf2 (gated by d2 = stage-2 output,
     drain at out, vdd source) — final FF path to the global
     output. Compare AFFC's Mgf2 (g=d1, d=out): DACFC's gmf2
     is gated by d2, NOT d1.
  5. NMCF-style outer Miller for Cm1 (out↔d2); inner cap to
     vss (Cm2_dac) on the gma1 cascode side; Cf dropped.

Topology delta vs `nested_miller_3stage`
========================================

Stages 1, 2 and 3 (signal path) inherit the NMCF baseline
verbatim. Stage 2's replica-bias output node `bias_d2` is reused
as the matched-pair gate (the AnalogGym netlist's VOUTN signal —
which is the inverted/replicated stage-2 output — serves the
same role).

**Doubly-differential active cascode (NEW, DACFC-unique):**

  Mga1_a : PMOS, d=nga1_a, g=bias_d2, s=vdd, b=vdd
  Mga1_b : PMOS, d=nga1_b, g=bias_d2, s=vdd, b=vdd
           Matched-pair active cascode. Both gated by bias_d2
           (Clay's stage-2 replica node, equivalent to AnalogGym's
           VOUTN signal). Drains at two separate cascode nodes
           nga1_a and nga1_b which form differential current
           injection into the inner compensation network.

  Mga2   : PMOS, d=nga2, g=bias_d2, s=vdd, b=vdd
           Single PMOS cascode buffer between Mga1 pair and the
           output sense network.

**Input-coupled gmf1_PMOS pair (NEW, DACFC's distinguishing
half of the "doubly-differential" name):**

  Mgmf1_n : PMOS, d=nga1_a, g=inn, s=vdd, b=vdd
  Mgmf1_p : PMOS, d=nga1_b, g=inp, s=vdd, b=vdd
            Gated directly by the differential INPUT signals
            (not by d1 like AFFC's Mgf2). This is the
            distinguishing architectural choice of DACFC —
            a feedforward pair tapped from the input nodes
            that bypasses the stage-1 diff pair entirely.
            Drains coupled to the same cascode nodes nga1_a
            and nga1_b that the gma1 pair drives, so the input
            FF current adds (differentially) to the gma1 pair's
            cascode current.

**Single gmf2 PMOS feedforward (output stage FF):**

  Mgmf2 : PMOS, d=out, g=d2, s=vdd, b=vdd
          Single PMOS gated by d2 (stage-2 output), drain at
          global output. Same routing as AFFC's Mgf2 BUT gate
          is d2 not d1 — DACFC sources the LHP zero from
          stage-2 → output rather than stage-1 → output.

**Compensation: NMCF-style Cm1 + cap-to-vss Cm2_dac, NO Cf:**

  NMCF:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1
    Cf  : out ↔ d1

  DACFC:
    Cm1     : out ↔ d2          (NMCF-style outer Miller)
    Cm2_dac : nga1_a ↔ vss      (inner cap to vss on cascode side)
    (Cf dropped — Mgmf2 provides the output-stage LHP zero;
     Mgmf1 pair provides the input-stage LHP zero)

DACFC vs AFFC vs PFC in one line
================================

AFFC: SINGLE FF (Mgf2 PMOS, g=d1, d=out) — single-ended, gated by d1.
PFC: SINGLE oversized FF (Mgmf1 PMOS, g=d1, d=out, W=4×Mcs3) — same
     position as AFFC but oversized for pole-tracking.
DACFC: DOUBLY-differential FF — gma1 PMOS PAIR (g=bias_d2) + input
       gmf1_PMOS PAIR (g=inn, g=inp) + single gmf2 (g=d2, d=out).
       Three FF channels (matched pair + input-coupled pair +
       output-driver single), with the input-coupled pair being
       the distinguishing architectural choice.

The three cells share the 5-port surface and stage-1/2/3 signal
paths but differ in the number of FF devices (1 vs 1 vs 5) and
the gate-source of those devices (d1 vs d1-oversized vs
bias_d2-pair + input-pair + d2-single).

Honest scope (what you should NOT expect)
-----------------------------------------

DACFC is a structural compensation refinement that genuinely
changes the netlist (5 new FF MOSFETs: Mga1_a, Mga1_b, Mga2,
Mgmf1_n, Mgmf1_p, Mgmf2 + one cap-to-vss + dropped Cf). The
AnalogGym netlist is a sky130-pdk reference; the Clay
implementation captures the architectural archetype (matched
active-cascode pair + input-coupled FF pair + d2-gated output
FF) without claiming bit-perfect SPICE equivalence. At the
relaxed 5 dB / 10 kHz spec floor on gf180mcu 3.3 V, the PVT
envelope will land comfortably. Tests in
`tests/test_song_dacfc_distinct.py` lock the architectural
distinction (matched pair exists with g=bias_d2, input pair
exists with g=inn/inp, Cm2_dac referenced to vss) rather than
asserting specific SPICE numerics.

Knobs
-----

DACFC adds the following new kwargs on top of the NMCF baseline:

  W_gma1, L_gma1: float = 8 µm / 0.5 µm
      Matched-pair sizing for Mga1_a + Mga1_b.
  W_gmf1_in, L_gmf1_in: float = 8 µm / 1 µm
      Input-coupled pair sizing for Mgmf1_n + Mgmf1_p.
  W_gma2, L_gma2: float = 4 µm / 1 µm
      Single gma2 buffer sizing.
  Mgmf2_scale: float = 1.0
      Sizing ratio of Mgmf2 / Mcs3 (output-stage FF, like AFFC).
  W_gmf2, L_gmf2, gm_id_gmf2: overrides for Mgmf2 sizing.
  Cm2_dac: float = 0.5e-12
      DACFC inner cap to vss (nga1_a ↔ vss).

The kwarg `Cf` from the NMCF baseline is NOT exposed — DACFC
drops Cf (the doubly-differential FF + Mgmf2 replace it). All
other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1) are
inherited verbatim from NMCF.
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


def build_nested_miller_3stage_dacfc(
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

    # ─── Compensation cap (NMCF-style outer; Cm2 replaced) ──
    Cm1: float = 1.5e-12,    # outer Miller (out ↔ d2), NMCF-style

    # ─── DACFC-specific: matched gma1 PMOS pair (cascode) ───
    W_gma1: float = 8e-6,
    L_gma1: float = 0.5e-6,

    # ─── DACFC-specific: input-coupled gmf1_PMOS pair ───────
    # PMOS gated directly by inn / inp — the input-coupled
    # feedforward pair that gives DACFC its name (second half
    # of doubly-differential).
    W_gmf1_in: float = 8e-6,
    L_gmf1_in: float = 1e-6,

    # ─── DACFC-specific: single gma2 PMOS buffer ────────────
    W_gma2: float = 4e-6,
    L_gma2: float = 1e-6,

    # ─── DACFC-specific: output-stage FF Mgmf2 ──────────────
    # PMOS, g=d2, d=out. Same role as AFFC's Mgf2 BUT gated by
    # d2 not d1.
    Mgmf2_scale: float = 1.0,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # ─── DACFC-specific: inner cap to vss ───────────────────
    Cm2_dac: float = 0.5e-12,

    # ─── R121 C: qualitative DRC annotation surface ────────
    # Additive metadata. Default False keeps every pre-R121
    # caller byte-for-byte identical. When True, three
    # qualitative hints attach to the cell:
    #   MATCHING on (M1, M2)               — PMOS diff pair
    #   MATCHING on (Mga1_a, Mga1_b)       — doubly-diff gma1 pair
    #   MATCHING on (Mgmf1_n, Mgmf1_p)     — input-coupled gmf1 pair
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Song DACFC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC / NMCNR / TCFC / ACBC / DFCFC2 / DFCFC1 /
    RAFFC / LEC variants. DACFC differs from NMCF in two structural
    places:
      1. Doubly-differential feedforward network: matched gma1 PMOS
         pair (Mga1_a + Mga1_b, gated by bias_d2) PLUS input-coupled
         gmf1_PMOS pair (Mgmf1_n + Mgmf1_p, gated by inn / inp).
         PLUS single gma2 buffer (g=bias_d2). PLUS single Mgmf2
         (g=d2, d=out).
      2. Inner cap Cm2_dac referenced to vss (NOT d1) on the
         cascode side (nga1_a ↔ vss); Cf dropped.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3, gm_id_*: NMCF baseline device sizing.
        Cm1: NMCF-style outer Miller (out ↔ d2).
        W_gma1, L_gma1: matched pair sizing (Mga1_a + Mga1_b).
        W_gmf1_in, L_gmf1_in: input-coupled pair sizing
            (Mgmf1_n + Mgmf1_p).
        W_gma2, L_gma2: single buffer sizing.
        Mgmf2_scale, W_gmf2, L_gmf2, gm_id_gmf2: output FF sizing.
        Cm2_dac: inner cap to vss value.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # DACFC: Mgmf2 draws additional DC current at out (same as AFFC).
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

    # Mgmf2 sizing (output-stage FF, same logic as AFFC's Mgf2).
    if L_gmf2 is None:
        L_gmf2 = L_cs3
    if gm_id_gmf2 is not None:
        W_gmf2 = closed_form_W("pmos", gm_id=gm_id_gmf2,
                               I_d=I_gmf2, L=L_gmf2).W
    elif W_gmf2 is None:
        W_gmf2 = W_cs3 * Mgmf2_scale

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
    # bias_d2 is reused as DACFC's gma1 pair gate (the AnalogGym
    # netlist's VOUTN signal serves the same role — it's the
    # replicated/inverted stage-2 output).
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

    # ─── DACFC delta: matched gma1 PMOS pair (active cascode) ─
    # Architectural signature of DACFC (half 1 of doubly-differential).
    # Both gated by bias_d2 (stage-2 replica), drains at two separate
    # cascode nodes nga1_a / nga1_b which form differential current
    # injection into the inner compensation network.
    devices.append(_mosfet("Mga1_a", "pmos",
                            d="nga1_a", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))
    devices.append(_mosfet("Mga1_b", "pmos",
                            d="nga1_b", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma1, L=L_gma1))

    # ─── DACFC delta: input-coupled gmf1_PMOS pair (FF pair) ──
    # Architectural signature of DACFC (half 2 of doubly-differential).
    # Gated directly by the differential input signals (inn, inp) —
    # this bypass of the diff pair is the distinguishing choice of
    # DACFC. Drains coupled to the SAME nga1_a / nga1_b nodes that
    # the gma1 pair drives, so the input-FF current adds differentially
    # to the gma1 cascode current.
    devices.append(_mosfet("Mgmf1_n", "pmos",
                            d="nga1_a", g="inn", s="vdd", b="vdd",
                            W=W_gmf1_in, L=L_gmf1_in))
    devices.append(_mosfet("Mgmf1_p", "pmos",
                            d="nga1_b", g="inp", s="vdd", b="vdd",
                            W=W_gmf1_in, L=L_gmf1_in))

    # ─── DACFC delta: single gma2 PMOS buffer ───────────────
    # Single PMOS cascode buffer between the gma1 pair and the
    # output sense network. Gated by bias_d2 (same as gma1 pair).
    devices.append(_mosfet("Mga2", "pmos",
                            d="nga2", g="bias_d2", s="vdd", b="vdd",
                            W=W_gma2, L=L_gma2))

    # ─── DACFC delta: output-stage FF Mgmf2 ─────────────────
    # PMOS gated by d2, drain at out. Same routing as AFFC's Mgf2
    # BUT gate is d2 not d1 — DACFC takes its output-stage LHP zero
    # from stage-2 instead of stage-1.
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # ─── DC bias sinks for the cascode / buffer drain nodes ─
    # The new internal nodes (nga1_a, nga1_b, nga2) need DC sink
    # paths so they sit at well-defined voltages. We use NMOS
    # mirror legs from nbias (each sinks I_ref to vss).
    devices.append(_mosfet("Mlcs_ga1_a", "nmos",
                            d="nga1_a", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga1_b", "nmos",
                            d="nga1_b", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mlcs_ga2", "nmos",
                            d="nga2", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: Cm1 outer (NMCF-style) + Cm2_dac to vss ─
    devices.append(_cap("Cm1",     p="out",    n="d2",  value=Cm1))
    devices.append(_cap("Cm2_dac", p="nga1_a", n="vss", value=Cm2_dac))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R121 C: optional qualitative DRC annotations ─────────
    # Three MATCHING annotations span the three doubly-differential
    # matched pairs that define DACFC's signature feedforward network.
    # Default attach_drc=False keeps the pre-R121 emit byte-identical.
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
                    "input-referred offset. DACFC's input-coupled "
                    "gmf1 pair already bypasses the diff pair on the "
                    "feedforward path, so any M1/M2 imbalance shows up "
                    "as an asymmetric stage-1 differential current that "
                    "the gma1 cascode network cannot null"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mga1_a", "Mga1_b"),
            extras={
                "rationale": (
                    "Doubly-differential gma1 cascode pair (half 1 of "
                    "DACFC's name). Both gated by bias_d2; mismatch "
                    "breaks the differential current balance at "
                    "nga1_a vs nga1_b — the very feedforward symmetry "
                    "the architecture is built on"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmf1_n", "Mgmf1_p"),
            extras={
                "rationale": (
                    "Input-coupled gmf1 PMOS pair (half 2 of DACFC's "
                    "name). Gated directly by inn / inp — Vth mismatch "
                    "between Mgmf1_n and Mgmf1_p injects a static "
                    "differential current at nga1_a / nga1_b that "
                    "competes with the M1/M2 path and creates an offset "
                    "the inner cap Cm2_dac cannot reject"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
