"""
Built-in template: nested_miller_3stage_clia.

Three-stage opamp with **Capacitor-Less Indirect Amplifier**
compensation (Tan CLIA) — an NMCF-family variant where the entire
Miller compensation network is replaced by an active current-mode
local impedance amplifier (CLIA) trio. NO Miller cap. NO Cm1. NO
Cm2. The compensation is purely active, current-mode, through
three matched gm cells that form a local-impedance feedback loop
around the stage-2 output node.

The "capacitor-less" name is literal: the SPICE netlist of a CLIA
cell does NOT contain any compensation cap in the inner-Miller
position. Pole-splitting is provided by the low-impedance local
current-mode loop instead of the standard Miller multiplication
of a passive cap. The only cap that remains is an outer
compensation cap from stage-3 output to an internal RC node
(architecturally distinct from NMCF's Cm1 which spans out↔d2 —
the CLIA outer cap lands on a CURRENT-mirror diode node, not the
direct stage-2 output gate).

Reference
=========

Tan_CLIA_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Tan_CLIA_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm70 net3  net050 VDDA VDDA pfet → Mgm21_PMOS (gm21 PMOS half,
                                       d=ngm21, g=d2 stage-2
                                       output)
  xm72 net5  net3   GNDA GNDA nfet → Mgm23_NMOS_b (gm23 NMOS,
                                       d=nclia_drv, g=ngm21 —
                                       gated by the gm21 PMOS
                                       drain, closes the
                                       current-mode local loop)
  xm71 net3  net3   GNDA GNDA nfet → Mgm23_NMOS_a (gm23 NMOS
                                       diode at ngm21 — the
                                       current-mirror diode in
                                       the local loop)
  xm68 VOUT  net050 VDDA VDDA pfet → Mgmf (active feedforward
                                       PMOS, g=d2, d=out — feeds
                                       the global output)
  C0 net2  GNDA   → Cclia_gnd (single outer compensation cap to
                                       vss, on the inner RC node)
  R0 net5  net2   → Rclia (nulling-R between the gm23 NMOS drain
                                       node nclia_drv and the
                                       cap node nclia_cap)

  NO compensation cap in the Miller position (out↔d2 or d2↔d1).
  This is the architectural signature of CLIA.

The CLIA architectural fingerprint:
  1. **gm21 PMOS** (Mgm21_PMOS, PMOS, d=ngm21, g=d2, s=vdd) —
     senses the stage-2 output and converts it to a current
     into the gm23 NMOS mirror diode.
  2. **gm23 NMOS diode** (Mgm23_NMOS_a, NMOS, d=g=ngm21, s=vss)
     — the diode side of the gm23 current mirror. Sets the local
     loop's reference current.
  3. **gm23 NMOS mirror** (Mgm23_NMOS_b, NMOS, d=nclia_drv,
     g=ngm21, s=vss) — the mirror side of gm23. Drives the
     current-mode local impedance back into the inner RC node.
     The trio (gm21 PMOS + gm23 NMOS diode + gm23 NMOS mirror)
     is the CURRENT-mode local impedance amplifier that replaces
     the Miller cap.
  4. **active FF Mgmf** (PMOS, d=out, g=d2, s=vdd) — feedforward
     PMOS at the global output. Note the gate is d2 (NOT d1 like
     AFFC's Mgf2), distinguishing CLIA's FF routing.
  5. ONE outer compensation cap `Cclia_gnd` to vss (NOT a Miller
     cap between out↔d2 — it's a cap from an internal RC node
     to vss).
  6. ONE nulling-R `Rclia` (between the gm23 NMOS mirror drain
     `nclia_drv` and the cap node `nclia_cap`).
  7. NMCF's Cm1, Cm2, Cf — ALL DROPPED. CLIA has zero
     Miller-position caps. This is the literal "capacitor-less"
     signature.

Topology delta vs `nested_miller_3stage`
========================================

Stages 1, 2, and 3 (signal path) inherit the NMCF baseline
verbatim. The deltas are entirely in the compensation network:

**gm21 PMOS Mgm21_PMOS (NEW, CLIA-unique):**

  Mgm21_PMOS : PMOS, d=ngm21, g=d2, s=vdd, b=vdd
               Senses stage-2 output (d2) and converts to a
               current into the gm23 NMOS mirror diode at ngm21.

**gm23 NMOS pair Mgm23_NMOS_a / Mgm23_NMOS_b (NEW, CLIA-unique):**

  Mgm23_NMOS_a : NMOS, d=ngm21, g=ngm21, s=vss, b=vss
                 Diode-connected at ngm21. Sets the local loop's
                 reference current and bias for the mirror side.
  Mgm23_NMOS_b : NMOS, d=nclia_drv, g=ngm21, s=vss, b=vss
                 Mirror side. Drives current-mode impedance back
                 into the inner RC node, completing the local
                 loop.

**Active feedforward Mgmf (NEW, CLIA-unique):**

  Mgmf : PMOS, d=out, g=d2, s=vdd, b=vdd
         Feedforward PMOS from stage-2 to global output. Note the
         gate is d2 (NOT d1 like AFFC's Mgf2 g=d1) — CLIA takes
         its output-stage LHP zero from stage-2, similar to DACFC.

**Outer compensation network (CLIA routing — NO Miller cap):**

  NMCF baseline:
    Cm1 : out ↔ d2     (Miller cap — DROPPED in CLIA)
    Cm2 : d2  ↔ d1     (Miller cap — DROPPED in CLIA)
    Cf  : out ↔ d1     (Feedforward cap — DROPPED in CLIA)

  CLIA routing:
    Cclia_gnd : nclia_cap ↔ vss   (cap to vss on inner RC node)
    Rclia     : nclia_drv ↔ nclia_cap   (nulling-R between gm23
                                          mirror drain and cap)
    (NO Miller cap between out↔d2, d2↔d1, or out↔d1.
     The CLIA local-impedance current-mode loop replaces all of
     them — this is the architectural meaning of "capacitor-less".)

CLIA vs LEC vs AZC in one line
==============================

LEC: local error amp Mlea + buffer Mleb + ONE cap-to-vss Clec_gnd
     + ONE nulling-R Rlec + Cm1 skip-over routing (d1↔out).
     LEC keeps an outer Miller cap (Cm1 d1↔out).
AZC: triple Mazc + quad-R mesh + triple cap (2 to vss, 1 outer
     Miller-like) — full charge-redistribution network. AZC keeps
     ONE outer Miller-like cap (Cazc0 out↔nazc_bias).
CLIA: gm21 PMOS + gm23 NMOS pair (current-mode local loop) + ONE
      cap-to-vss Cclia_gnd + ONE nulling-R Rclia + active FF Mgmf
      (g=d2). **NO Miller-position cap at all** — the inner cap
      sits on an internal RC node to vss, not the output node.

The "capacitor-less" name applies to the Miller position: CLIA
literally has ZERO compensation caps between out↔d2, d2↔d1, or
out↔d1. The single Cclia_gnd cap is to vss on an internal RC
node, which by convention is NOT a Miller cap. This is the
architecturally most distinct of all NMCF-family variants.

Honest scope (what you should NOT expect)
-----------------------------------------

CLIA is a fundamentally different compensation philosophy from
the Miller-cap-based NMCF family — the absence of any
out↔d2 / d2↔d1 / out↔d1 cap means the pole-splitting mechanism
is entirely different (active current-mode local impedance vs
passive cap multiplication). At the relaxed 5 dB / 10 kHz spec
floor on gf180mcu 3.3 V, the resulting GBW and phase margin will
land in a completely different region than NMCF baseline —
expect very different absolute numbers (likely much wider GBW
because there's no Miller cap to slow the loop, possibly worse
phase margin if Cclia_gnd alone can't tame the higher poles).
The PVT envelope may also be wider because the active loop's
gain depends on Mgm21_PMOS / Mgm23_NMOS gm_id tracking. Tests in
`tests/test_tan_clia_distinct.py` lock the architectural
distinction (gm21+gm23 trio exists, NO Cm1/Cm2/Cf, only
Cclia_gnd-to-vss + Rclia) rather than asserting specific SPICE
numerics.

Knobs
-----

CLIA adds the following new kwargs on top of the NMCF baseline:

  W_gm21, L_gm21: float = 8 µm / 1 µm   (gm21 PMOS sizing)
  W_gm23, L_gm23: float = 8 µm / 1 µm   (gm23 NMOS pair sizing)
  Mgmf_scale: float = 1.0
      Ratio of Mgmf W to Mcs3 W.
  W_gmf, L_gmf, gm_id_gmf: optional overrides for Mgmf.
  Cclia_gnd: float = 1.0e-12   (cap-to-vss on inner RC node)
  Rclia:     float = 5e3        (nulling-R between gm23 drain
                                  and cap node)

The kwargs `Cm1`, `Cm2`, `Cf` from the NMCF baseline are NOT
exposed — CLIA drops ALL three (the capacitor-less architecture
forbids any Miller-position cap). All other knobs (I_ref, W_bp,
L_bp, W_bn, L_bn, N_tail, K_cs3_load, W_diff, L_diff, W_load1,
L_load1, W_cs2, L_cs2, W_cs3, L_cs3, gm_id_diff, gm_id_load1,
gm_id_stage2, gm_id_stage3) are inherited verbatim from NMCF.
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


def build_nested_miller_3stage_clia(
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

    # ─── CLIA-specific: gm21 PMOS sizing ────────────────────
    W_gm21: float = 8e-6,
    L_gm21: float = 1e-6,

    # ─── CLIA-specific: gm23 NMOS pair sizing ───────────────
    W_gm23: float = 8e-6,
    L_gm23: float = 1e-6,

    # ─── CLIA-specific: active FF Mgmf ──────────────────────
    Mgmf_scale: float = 1.0,
    W_gmf: float | None = None,
    L_gmf: float | None = None,
    gm_id_gmf: float | None = None,

    # ─── CLIA-specific: compensation network (NO Miller cap) ─
    Cclia_gnd: float = 1.0e-12,   # cap to vss on inner RC node
    Rclia:     float = 5e3,       # nulling-R between gm23 drain
                                  #   and cap node

    # ─── R114 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # CLIA 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node from supply-coupled
    #     substrate noise.
    #   * MATCHING on the CLIA gm23 NMOS mirror pair (Mgm23_NMOS_a diode
    #     leg + Mgm23_NMOS_b mirror leg). This pair forms the active
    #     current-mode local-loop that REPLACES the Miller cap in CLIA
    #     (Tan §III); mismatch in the gm23 pair directly maps to local-
    #     loop gain error and degrades phase-margin. 1:1 mirror — pure
    #     MATCHING (not CURRENT_MIRROR_MATCHING which targets 1:N
    #     unit-cell arrays). Common-centroid required.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Tan CLIA 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and all NMCF-family variants. CLIA differs from NMCF FUNDAMENTALLY:
    it has NO Miller cap (no out↔d2, no d2↔d1, no out↔d1 — all dropped).
    The Miller multiplication is replaced by an active current-mode
    local impedance amplifier formed by gm21 PMOS + gm23 NMOS mirror
    pair, plus a single cap-to-vss + nulling-R on an internal RC node.

    Structural deltas vs NMCF:
      1. gm21 PMOS (Mgm21_PMOS, g=d2, d=ngm21) — senses stage-2 output.
      2. gm23 NMOS pair (Mgm23_NMOS_a diode + Mgm23_NMOS_b mirror) —
         current-mode local loop.
      3. Active feedforward Mgmf (PMOS, g=d2, d=out) — provides LHP
         zero through FF leg.
      4. Single cap Cclia_gnd to vss (NOT Miller position) + single
         nulling-R Rclia. NO Cm1/Cm2/Cf.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3, gm_id_*: NMCF baseline device sizing.
        W_gm21, L_gm21: gm21 PMOS sizing.
        W_gm23, L_gm23: gm23 NMOS pair sizing.
        Mgmf_scale, W_gmf, L_gmf, gm_id_gmf: active FF sizing.
        Cclia_gnd: cap-to-vss on inner RC node (NOT a Miller cap).
        Rclia: nulling-R between gm23 drain and cap node.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # CLIA: Mgmf draws additional DC current at out (same as AFFC).
    I_gmf = Mgmf_scale * K_cs3_load * I_ref
    K_cs3_load_total = int(round(K_cs3_load * (1 + Mgmf_scale)))
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

    # Mgmf sizing.
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

    # ─── CLIA delta: gm21 PMOS (senses stage-2 output) ──────
    # Architectural signature of CLIA (part 1 of trio). PMOS gated by
    # d2 (stage-2 output), drain at ngm21 (the gm23 NMOS mirror diode
    # node). Converts d2 to a current into the local loop.
    devices.append(_mosfet("Mgm21_PMOS", "pmos",
                            d="ngm21", g="d2", s="vdd", b="vdd",
                            W=W_gm21, L=L_gm21))

    # ─── CLIA delta: gm23 NMOS diode ────────────────────────
    # Architectural signature of CLIA (part 2 of trio). Diode-connected
    # NMOS at ngm21 — sets the local loop's reference current and
    # provides the diode side of the gm23 current mirror.
    devices.append(_mosfet("Mgm23_NMOS_a", "nmos",
                            d="ngm21", g="ngm21", s="vss", b="vss",
                            W=W_gm23, L=L_gm23))

    # ─── CLIA delta: gm23 NMOS mirror ───────────────────────
    # Architectural signature of CLIA (part 3 of trio). Mirror side of
    # gm23 — drain at nclia_drv (the inner RC node before the nulling
    # resistor), gate at ngm21 (mirrors the gm23 diode current).
    # This is the current-mode local impedance back to the inner RC
    # node, completing the local loop without any Miller cap.
    devices.append(_mosfet("Mgm23_NMOS_b", "nmos",
                            d="nclia_drv", g="ngm21", s="vss", b="vss",
                            W=W_gm23, L=L_gm23))

    # ─── CLIA delta: active feedforward Mgmf ────────────────
    # Active FF from stage-2 to output. PMOS gated by d2 (NOT d1 like
    # AFFC's Mgf2) — CLIA's distinct FF gate.
    devices.append(_mosfet("Mgmf", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── CLIA compensation network: NO Miller cap ───────────
    # Cclia_gnd: cap-to-vss on inner RC node. NOT a Miller cap (does
    # not span out↔d2 or d2↔d1 or out↔d1). This is the only cap in
    # the compensation network — hence "capacitor-less" for the
    # Miller position.
    devices.append(_cap("Cclia_gnd", p="nclia_cap", n="vss", value=Cclia_gnd))
    # Rclia: nulling-R between gm23 mirror drain (nclia_drv) and the
    # cap node (nclia_cap). Closes the RC nulling path.
    devices.append(_res("Rclia", p="nclia_drv", n="nclia_cap", value=Rclia))

    # DC bias path for nclia_cap so it doesn't float. Tie to nbias via
    # a small NMOS leg (sinks I_ref to vss).
    devices.append(_mosfet("Mbn_clia_cap", "nmos",
                            d="nclia_cap", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

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
                    "tied to vdd shields the high-Z d1 node from "
                    "supply-coupled substrate noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgm23_NMOS_a", "Mgm23_NMOS_b"),
            extras={
                "rationale": (
                    "CLIA gm23 NMOS mirror pair — Mgm23_NMOS_a "
                    "(diode) / Mgm23_NMOS_b (mirror) form the "
                    "active current-mode local-loop that REPLACES "
                    "the Miller cap (Tan §III); mismatch maps to "
                    "loop-gain error and degrades phase-margin; "
                    "1:1 ratio, common-centroid required"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
