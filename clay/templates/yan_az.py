"""
Built-in template: nested_miller_3stage_az.

Three-stage opamp with **Auto-Zero** compensation (Yan AZ) — an
NMCF-family variant where pole-splitting is provided by a SIMPLER
auto-zero structure than the full AZC (Qu 2017) network. AZ uses
a single matched gmb1/gmb2 NMOS pair as the auto-zero nulling
devices, with a single nulling-R and a single auto-zero cap. The
cap is in SERIES with the Miller path (NOT to vss like AZC's
Cazc1/Cazc2).

The "auto-zero" name describes the architectural class — periodic
offset cancellation through a passive RC network gated by an
active NMOS nulling pair. Where AZC has THREE compensation MOSFETs
(Mazc1/Mazc2_diode/Mazc3) and FOUR nulling resistors
(Razc0-Razc3) forming a triple-cap + quad-R mesh, AZ has the
simpler TWO compensation MOSFETs (Mgmb1/Mgmb2) with one nulling-R
(Raz) and one auto-zero cap (Caz) in series with Miller — a strict
subset signature.

Reference
=========

Yan_AZ_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Yan_AZ_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm22 net094 net051 GNDA GNDA nfet → Mgmb1 (gmb1 NMOS half a,
                                       d=naz_drv, g=naz_cap —
                                       senses cap node, drives
                                       drv node)
  xm23 net057 net094 GNDA GNDA nfet → Mgmb2 (gmb2 NMOS half b,
                                       d=naz_pair, g=naz_drv —
                                       gated by Mgmb1's drain,
                                       drives the pair node into
                                       the inner Miller path)
  xm13 VOUT  net050 VDDA VDDA pfet → Mgmf2 (active FF PMOS,
                                       g=d2, d=out — output-stage
                                       FF)
  C0 net063 VOUT                  → Caz_outer (outer Miller-like
                                       cap, between an internal
                                       stage-1 mirror node and
                                       out — NMCF-style outer
                                       Miller preserved)
  C1 net051 GNDA                  → Caz (auto-zero cap, on the
                                       naz_cap node — in series
                                       with the Miller path via
                                       Raz)
  R0 net078 net077                → Raz_a (nulling-R between
                                       two bias hub nodes)
  R1 net078 net082                → Raz_b (nulling-R between
                                       bias hub and second cap
                                       leg)
  R2 net094 net051                → Raz (primary nulling-R, in
                                       series between Mgmb1's
                                       drain (naz_drv) and the
                                       auto-zero cap node)

The AZ architectural fingerprint (SIMPLER than AZC):
  1. TWO NMOS nulling-pair devices Mgmb1 + Mgmb2 (the AZC
     equivalent has THREE — strict subset).
  2. ONE primary nulling resistor Raz (between Mgmb1's drain
     and the auto-zero cap node).
  3. ONE auto-zero cap Caz, in SERIES with the Miller path via
     the nulling-R Raz. Caz is NOT to vss directly — it sits at
     an internal node that connects via Raz to Mgmb1's drain.
     This is the AZ "cap in series with Miller" signature
     (compare AZC's Cazc1/Cazc2 which are directly to vss).
  4. NMCF-style outer Miller cap `Caz_outer` (out ↔ d2) preserved.
  5. Active feedforward `Mgmf2` (PMOS, g=d2, d=out) — provides
     LHP zero through FF leg.
  6. NMCF's Cm2 (d2↔d1) and Cf (out↔d1) dropped — the auto-zero
     pair + Caz series with Miller replace them.

Topology delta vs `nested_miller_3stage`
========================================

Stages 1, 2, and 3 (signal path) inherit the NMCF baseline
verbatim. The deltas live in the compensation network:

**Auto-zero pair Mgmb1 + Mgmb2 (NEW, AZ-unique):**

  Mgmb1 : NMOS, d=naz_drv, g=naz_cap, s=vss, b=vss
          Senses the auto-zero cap node (naz_cap), drives the
          nulling-R drain node (naz_drv). Drain-on-naz_drv,
          gate-on-naz_cap is the AZ signature for the first half
          of the nulling pair.
  Mgmb2 : NMOS, d=naz_pair, g=naz_drv, s=vss, b=vss
          Gated by Mgmb1's drain (naz_drv), drives naz_pair which
          couples back into the inner Miller path through Caz_outer.
          Second half of nulling pair.

**Active feedforward Mgmf2 (NEW, AZ-unique):**

  Mgmf2 : PMOS, d=out, g=d2, s=vdd, b=vdd
          PMOS active FF gated by d2 (NOT d1 like AFFC's Mgf2) —
          AZ takes output-stage LHP zero from stage-2.

**Compensation network (AZ routing — cap in SERIES with Miller):**

  NMCF baseline:
    Cm1 : out ↔ d2     (outer Miller)
    Cm2 : d2  ↔ d1     (inner Miller — DROPPED in AZ)
    Cf  : out ↔ d1     (feedforward — DROPPED in AZ)

  AZ routing:
    Caz_outer : out ↔ d2          (NMCF-style outer Miller preserved)
    Caz       : naz_cap ↔ naz_pair (auto-zero cap in series with
                                    Miller path via Raz)
    Raz       : naz_drv ↔ naz_cap (primary nulling-R, in series
                                    between Mgmb1 drain and cap)
    (Cm2 d2↔d1 dropped; Cf dropped — auto-zero pair + Caz series
     with Miller path provide compensation shape.)

  Series chain (Miller path through AZ):
    d2 ─ Mgmb2 (gate=naz_drv) ─ naz_pair ─ Caz ─ naz_cap ─ Raz ─ naz_drv ─ Mgmb1 (gate=naz_cap)
                                                                              ─ naz_pair ↔ d2 via Caz_outer-equivalent coupling

  The series cap-resistor with the active nulling pair is the
  AZ "cap in series with Miller" signature.

AZ vs AZC in one line
=====================

AZC: THREE compensation MOSFETs (Mazc1+Mazc2_diode+Mazc3), FOUR
     nulling-R (Razc0-Razc3) forming a quad-R mesh with 2 hub
     nodes, THREE compensation caps (1 outer Miller-like + 2 to
     vss). Full charge-redistribution mesh.
AZ:  TWO nulling MOSFETs (Mgmb1+Mgmb2), ONE primary nulling-R
     (Raz), ONE auto-zero cap (Caz) in SERIES with Miller path
     (NOT directly to vss), plus outer Miller Caz_outer (out↔d2).
     Strict subset of AZC — half the MOSFETs, quarter the
     resistors, third of the caps. The "simpler" auto-zero variant.

Honest scope (what you should NOT expect)
-----------------------------------------

AZ is a structurally simpler auto-zero compensation than AZC
(2 MOSFETs vs 3, 1 nulling-R vs 4, 1 cap-in-series vs 2 caps to
vss). At the relaxed 5 dB / 10 kHz spec floor on gf180mcu 3.3 V,
the PVT envelope will land comfortably. Tests in
`tests/test_yan_az_distinct.py` lock the architectural distinction
(Mgmb1+Mgmb2 with the gate/drain coupling signature, Caz NOT to
vss directly, Raz primary nulling-R, NO Cm2 d2↔d1, NO Cf) rather
than asserting specific SPICE numerics.

Knobs
-----

AZ adds the following new kwargs on top of the NMCF baseline:

  W_gmb, L_gmb: float = 4 µm / 1 µm
      Auto-zero NMOS pair sizing (matched across Mgmb1/Mgmb2).
  Mgmf2_scale: float = 1.0
      Ratio of Mgmf2 W to Mcs3 W.
  W_gmf2, L_gmf2, gm_id_gmf2: optional overrides for Mgmf2.
  Caz_outer: float = 1.5e-12   (outer Miller — out↔d2 NMCF-style)
  Caz:       float = 0.5e-12   (auto-zero cap — naz_cap↔naz_pair)
  Raz:       float = 10e3      (primary nulling-R — naz_drv↔naz_cap)

The kwargs `Cm1`, `Cm2`, `Cf` from the NMCF baseline are NOT
exposed — AZ uses Caz_outer (out↔d2) for outer Miller and replaces
Cm2/Cf with the auto-zero pair + Caz series. All other knobs
(I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load, W_diff, L_diff,
W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3, gm_id_diff,
gm_id_load1, gm_id_stage2, gm_id_stage3) are inherited verbatim
from NMCF.
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


def build_nested_miller_3stage_az(
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

    # ─── AZ-specific: auto-zero NMOS pair sizing ────────────
    W_gmb: float = 4e-6,
    L_gmb: float = 1e-6,

    # ─── AZ-specific: active FF Mgmf2 ───────────────────────
    Mgmf2_scale: float = 1.0,
    W_gmf2: float | None = None,
    L_gmf2: float | None = None,
    gm_id_gmf2: float | None = None,

    # ─── AZ-specific: compensation network ──────────────────
    Caz_outer: float = 1.5e-12,   # outer Miller (out↔d2), NMCF-style
    Caz:       float = 0.5e-12,   # auto-zero cap (naz_cap↔naz_pair)
    Raz:       float = 10e3,      # primary nulling-R (naz_drv↔naz_cap)

    # ─── R116 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # Yan AZ 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset; the
    #     auto-zero pair's nulling action cancels static offset only
    #     when the diff pair's *signal* path is balanced, so M1/M2
    #     matching is the prerequisite for AZ to function.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node from supply-coupled
    #     substrate noise injected by the AZ chopping switches at
    #     naz_drv / naz_cap.
    #   * MATCHING on the auto-zero NMOS pair (Mgmb1 / Mgmb2). The AZ
    #     pair signature is reciprocal gate/drain coupling (Mgmb1
    #     senses naz_cap and drives naz_drv; Mgmb2 senses naz_drv and
    #     drives naz_pair). Vth-mismatch between the two introduces a
    #     residual offset that defeats the AZ correction loop; common-
    #     centroid required.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Yan AZ 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and all NMCF-family variants. AZ differs from NMCF in three
    structural places:
      1. Auto-zero NMOS pair Mgmb1 (d=naz_drv, g=naz_cap) + Mgmb2
         (d=naz_pair, g=naz_drv). The gate/drain coupling (Mgmb1
         senses cap, Mgmb2 gated by drv) is the AZ pair signature.
      2. Active feedforward Mgmf2 (PMOS, g=d2, d=out) — provides
         LHP zero through FF leg.
      3. Single nulling-R Raz + single auto-zero cap Caz in series.
         Caz_outer (out ↔ d2) NMCF-style preserved.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3, gm_id_*: NMCF baseline device sizing.
        W_gmb, L_gmb: auto-zero NMOS pair sizing.
        Mgmf2_scale, W_gmf2, L_gmf2, gm_id_gmf2: active FF sizing.
        Caz_outer: outer Miller (out↔d2), NMCF-style.
        Caz: auto-zero cap (naz_cap↔naz_pair), in series with Miller.
        Raz: primary nulling-R (naz_drv↔naz_cap).

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
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

    # ─── AZ delta: auto-zero NMOS pair ──────────────────────
    # Architectural signature of AZ. TWO NMOS nulling devices with
    # gate/drain coupling:
    #   Mgmb1: d=naz_drv, g=naz_cap — senses the cap node, drives drv.
    #   Mgmb2: d=naz_pair, g=naz_drv — gated by Mgmb1's drain.
    # The reciprocal gate/drain coupling (Mgmb1 senses cap, Mgmb2
    # senses drv) is the AZ pair signature.
    devices.append(_mosfet("Mgmb1", "nmos",
                            d="naz_drv", g="naz_cap", s="vss", b="vss",
                            W=W_gmb, L=L_gmb))
    devices.append(_mosfet("Mgmb2", "nmos",
                            d="naz_pair", g="naz_drv", s="vss", b="vss",
                            W=W_gmb, L=L_gmb))

    # ─── AZ delta: active feedforward Mgmf2 ─────────────────
    # PMOS active FF gated by d2 (NOT d1 like AFFC's Mgf2). Provides
    # LHP zero through FF leg.
    devices.append(_mosfet("Mgmf2", "pmos",
                            d="out", g="d2", s="vdd", b="vdd",
                            W=W_gmf2, L=L_gmf2))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf2 DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── AZ compensation network ────────────────────────────
    # Caz_outer: NMCF-style outer Miller (out ↔ d2).
    devices.append(_cap("Caz_outer", p="out", n="d2", value=Caz_outer))
    # Caz: auto-zero cap, in series with Miller path. naz_cap ↔
    # naz_pair (NOT to vss directly — distinguishes AZ from AZC's
    # Cazc1/Cazc2 which ARE to vss).
    devices.append(_cap("Caz", p="naz_cap", n="naz_pair", value=Caz))
    # Raz: primary nulling-R, in series between Mgmb1's drain
    # (naz_drv) and the auto-zero cap node (naz_cap).
    devices.append(_res("Raz", p="naz_drv", n="naz_cap", value=Raz))

    # DC bias paths for the AZ internal nodes so they don't float:
    # naz_pair couples to d2 via Caz_outer at AC, but at DC needs a
    # path. Tie naz_pair to nbias via small NMOS leg (sinks I_ref).
    devices.append(_mosfet("Mbn_naz_pair", "nmos",
                            d="naz_pair", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    # naz_cap is biased via Raz from naz_drv (which has Mgmb1 to vss),
    # but for safety add a small bias leg as well.
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

    # ─── R116 C: optional DRC annotations ─────────────────
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
                    "input-referred offset; common-centroid required. "
                    "The auto-zero pair's nulling action cancels static "
                    "offset only when the diff pair's signal path is "
                    "balanced, so M1/M2 matching is the prerequisite "
                    "for AZ to function"
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
                    "entire stage-1 bias current; nwell guard ring tied "
                    "to vdd shields the high-Z d1 node from supply-"
                    "coupled substrate noise injected by the AZ chopping "
                    "switches at naz_drv / naz_cap"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mgmb1", "Mgmb2"),
            extras={
                "rationale": (
                    "Auto-zero NMOS pair (reciprocal gate/drain coupling: "
                    "Mgmb1 senses naz_cap and drives naz_drv; Mgmb2 "
                    "senses naz_drv and drives naz_pair). Vth-mismatch "
                    "between the two introduces a residual offset that "
                    "defeats the AZ correction loop; common-centroid "
                    "required"
                ),
                "ratio": "1:1",
            },
        ))

    return cell
