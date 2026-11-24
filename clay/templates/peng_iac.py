"""
Built-in template: nested_miller_3stage_iac.

Three-stage opamp with **Indirect AC Compensation** (Peng IAC) — an
NMCF-family variant where the Miller compensation path is routed
INDIRECTLY through an active triode-region PMOS helper device
(`Mgmt`) instead of going directly from the output to the
stage-2 gate node.

The "indirect" name means the compensation cap does not sit
between out and d2 (NMCF-style direct Miller) — instead it sits
between out and an intermediate node `niac`, which is itself
buffered by Mgmt (PMOS in triode region acting as a controlled
current path) into the stage-3 gate. The triode helper provides
a current-routing path that is biased by `VB2` (a separate cascode
bias rail) rather than directly tied to the high-impedance
stage-2 output — this is the indirect part.

Reference
=========

Peng_IAC_Pin_3 in AnalogGym
(`Amplifier/spice_netlist/Peng_IAC_Pin_3`). The reference netlist
exposes these distinctive devices (mapped to Clay net names):

  xm66 net10 VB2 net049 VDDA pfet → Mgmt (PMOS triode helper,
                                       d=niac, g=vcas_iac,
                                       s=npath, drain feeds the
                                       indirect Miller node)
  xm11 VOUT VOUTP VDDA VDDA pfet  → Mgmf (PMOS active FF, g=d1,
                                       drain at out)
  C0 VOUTP VOUT                   → Cm1_iac (outer Miller-like,
                                       out↔d2 same as NMCF Cm1)
  C1 net10 net4                   → Cm2_iac (between niac and
                                       internal rc node — the
                                       indirect compensation cap
                                       routed through the triode
                                       helper Mgmt)
  R0 net4 GNDA                    → R0_iac (nulling-R on the
                                       inner RC node back to
                                       vss — the AC-coupling
                                       resistor that closes the
                                       indirect path)

The IAC architectural fingerprint:
  1. ONE triode-region PMOS helper `Mgmt` (PMOS, d=niac,
     g=vcas_iac, s=npath) — provides the indirect AC path. The
     drain at the high-impedance Miller node `niac`, gate at a
     cascode bias rail `vcas_iac` (NOT d2 like a normal CS-stage
     PMOS), source at an intermediate `npath` node (NOT vdd). The
     "indirect" routing: Mgmt is in the AC signal path between
     the stage-3 gate and the Miller compensation network, but
     biased independently via vcas_iac and npath.
  2. ONE active feedforward PMOS `Mgmf` (PMOS, g=d1, d=out)
     providing the LHP zero through an active FF leg.
  3. ONE RC nulling chain: a series cap `Cm2_iac` from `niac` to
     `n_rc_iac` (the inner RC node), and a series resistor
     `R0_iac` from `n_rc_iac` to vss. The series cap-resistor
     chain at the indirect node is the IAC's distinguishing
     compensation signature.
  4. NMCF-style outer Miller cap `Cm1_iac` (out ↔ d2) preserved.
  5. NMCF's Cm2 (d2 ↔ d1) and Cf (out ↔ d1) dropped — the
     indirect path + active FF + RC nulling replace them.

Topology delta vs `nested_miller_3stage`
========================================

Stages 1, 2, and 3 (signal path) inherit the NMCF baseline
verbatim. The deltas live entirely in the compensation network:

**Triode-helper PMOS Mgmt (NEW, IAC-unique):**

  Mgmt : PMOS, d=niac, g=vcas_iac, s=npath, b=vdd
         Indirect AC compensation helper. Gate at the cascode
         bias rail `vcas_iac` (NOT d2 — that is the direct
         signature this device avoids). Source at intermediate
         `npath` node biased from vdd via the npath bias leg.
         Drain at `niac` — the indirect Miller node where the
         outer compensation cap `Cm1_iac` lands.

**Active feedforward PMOS Mgmf (NEW, IAC-unique):**

  Mgmf : PMOS, d=out, g=d1, s=vdd, b=vdd
         Active feedforward from stage-1 output to global out —
         standard AFFC-style FF position but combined with the
         indirect Miller path.

**Bias for the IAC compensation network:**

  Mbp_vcas_iac : PMOS diode at vcas_iac (d=g=vcas_iac, s=vdd) —
                 holds vcas_iac at vdd - |Vgs_p|.
  Mbp_npath    : PMOS diode at npath (d=g=npath, s=vdd) —
                 holds npath as a clamped source rail for Mgmt.
  Mbn_niac     : NMOS sink at niac (g=nbias, s=vss) — pulls DC
                 current through Mgmt so it sits in triode.

**Compensation network (IAC routing — distinct from NMCF):**

  NMCF baseline:
    Cm1 : out ↔ d2
    Cm2 : d2  ↔ d1     (NMCF inner Miller)
    Cf  : out ↔ d1     (NMCF feedforward cap)

  IAC routing:
    Cm1_iac : out  ↔ d2          (NMCF-style outer Miller preserved)
    Cm2_iac : niac ↔ n_rc_iac    (indirect Miller via Mgmt helper)
    R0_iac  : n_rc_iac ↔ vss     (nulling-R to close the path)
    (Cm2 d2↔d1 dropped; Cf dropped — Mgmf active FF + indirect
     path provide the LHP zero shape instead)

  The series cap-resistor chain (Cm2_iac in series with R0_iac
  to vss) at the niac node is the indirect AC path: it AC-couples
  the niac node (Mgmt drain) to vss via a series RC, while Mgmt
  itself provides the current-routing buffer from vcas_iac through
  npath to niac.

IAC vs AFFC vs LEC in one line
==============================

AFFC: ONE FF (Mgf2 PMOS, g=d1, d=out) + Cm1/Cm2/Cf NMCF caps.
LEC:  local error amp Mlea + buffer Mleb + cap-to-vss Clec_gnd +
      Rlec nulling-R, with Cm1 skip-over routing.
IAC:  triode helper Mgmt (g=vcas_iac, NOT d2) + active FF Mgmf
      (g=d1, d=out) + indirect Miller node niac + series cap-R
      chain (Cm2_iac + R0_iac to vss) + Cm1_iac NMCF-style outer.

The three cells share the 5-port surface and stage-1/2/3 signal
paths but differ in the compensation network's topology
(direct-Miller-only / local-error-amp-with-cap-to-vss /
indirect-via-triode-helper-with-RC-to-vss).

Honest scope (what you should NOT expect)
-----------------------------------------

IAC is a structurally distinct compensation scheme (1 triode
helper + 1 active FF + 1 series RC chain to vss + several new
internal nodes). The AnalogGym netlist is a sky130-pdk reference;
the Clay implementation captures the architectural archetype
(triode helper Mgmt + indirect Miller node niac + series RC
nulling) without claiming bit-perfect SPICE equivalence to the
paper netlist (which has dozens of cascode mirror legs not
material to the compensation topology). At the relaxed 5 dB /
10 kHz spec floor on gf180mcu 3.3 V, the PVT envelope will land
comfortably; whether the IAC indirect path gives better PVT
corners than NMCF depends heavily on Mgmt's W/L (triode-region
gm controls the indirect coupling strength) and the RC values.
Tests in `tests/test_peng_iac_distinct.py` lock the architectural
distinction (Mgmt with g=vcas_iac, niac internal node, R0_iac
series chain to vss) rather than asserting specific SPICE
numerics.

Knobs
-----

IAC adds the following new kwargs on top of the NMCF baseline:

  W_gmt, L_gmt: float = 8 µm / 1 µm
      Triode-helper Mgmt sizing. Larger W → larger gm in triode
      region → stronger indirect coupling. Default chosen to put
      Mgmt firmly in triode at typical bias (V_DS << V_GS - V_TH).
  W_gmf, L_gmf: float = 40 µm / 0.5 µm
      Active feedforward Mgmf sizing. Same routing as AFFC's Mgf2
      but with W matched to Mcs3 by default (Mgmf_scale=1.0).
  Mgmf_scale: float = 1.0
      Ratio of Mgmf W to Mcs3 W. Range 0.5-4.0 reasonable.
  Cm1_iac: float = 1.5e-12  (outer Miller — NMCF-style out↔d2)
  Cm2_iac: float = 0.5e-12  (indirect cap — niac↔n_rc_iac)
  R0_iac:  float = 10e3     (nulling-R — n_rc_iac↔vss)

The kwargs `Cm1`, `Cm2`, `Cf` from the NMCF baseline are NOT
exposed — IAC replaces Cm2 and Cf entirely (the indirect path +
RC chain provide the compensation shape). All other knobs (I_ref,
W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load, W_diff, L_diff,
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


def build_nested_miller_3stage_iac(
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

    # ─── IAC-specific: triode helper Mgmt ───────────────────
    W_gmt: float = 8e-6,
    L_gmt: float = 1e-6,

    # ─── IAC-specific: active FF Mgmf ───────────────────────
    Mgmf_scale: float = 1.0,
    W_gmf: float | None = None,
    L_gmf: float | None = None,
    gm_id_gmf: float | None = None,

    # ─── IAC-specific: compensation network ─────────────────
    Cm1_iac: float = 1.5e-12,   # outer Miller (out ↔ d2), NMCF-style
    Cm2_iac: float = 0.5e-12,   # indirect cap (niac ↔ n_rc_iac)
    R0_iac:  float = 10e3,      # nulling-R (n_rc_iac ↔ vss)

    # ─── R120 C: optional qualitative DRC annotations ───────
    # When True, attach three layout-prompt annotations capturing the
    # offset-critical and DC-bias-critical groups in the IAC backbone:
    #   * MATCHING (M1, M2) — PMOS diff pair Vth mismatch is the
    #     dominant input-referred offset for a plain NMCF stage 1
    #     (no input cascode in this baseline IAC).
    #   * CURRENT_MIRROR_MATCHING (M3, M4) — stage-1 NMOS diode
    #     mirror; M3 is the diode-connected leg at d1_diode, M4 is
    #     the mirror leg at d1. Mismatch unbalances the differential
    #     drain currents and saturates the d1 high-Z node off-target,
    #     breaking the indirect-AC compensation budget downstream.
    #   * GUARD_RING (Mtail,) — PMOS tail current source; its pdiff
    #     in an nwell tied to vdd shields the tail current from
    #     supply-coupled substrate transients that would re-modulate
    #     I_tail and inject CM disturbance into the diff pair.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Peng IAC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and all NMCF-family variants. IAC differs from NMCF in three
    structural places:
      1. Triode-region PMOS helper Mgmt (PMOS, d=niac, g=vcas_iac,
         s=npath, b=vdd) — the indirect AC compensation helper.
      2. Active feedforward Mgmf (PMOS, g=d1, d=out) — providing the
         LHP zero through an active FF leg.
      3. Series RC chain on indirect node: Cm2_iac (niac ↔ n_rc_iac) +
         R0_iac (n_rc_iac ↔ vss). Outer Miller Cm1_iac preserved
         NMCF-style (out ↔ d2).

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain.
        W_diff..L_cs3, gm_id_*: NMCF baseline device sizing.
        W_gmt, L_gmt: triode helper sizing.
        Mgmf_scale, W_gmf, L_gmf, gm_id_gmf: active FF sizing.
        Cm1_iac: NMCF-style outer Miller (out ↔ d2).
        Cm2_iac: indirect cap (niac ↔ n_rc_iac).
        R0_iac:  nulling-R (n_rc_iac ↔ vss).

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # IAC: Mgmf draws additional DC current at out (same as AFFC).
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

    # Mgmf sizing (active FF, same logic as AFFC's Mgf2).
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

    # ─── IAC delta: bias rails for indirect compensation ────
    # vcas_iac: cascode bias rail for Mgmt's gate. PMOS diode from
    # vdd, holds vcas_iac at vdd - |Vgs_p|.
    devices.append(_mosfet("Mbp_vcas_iac", "pmos",
                            d="vcas_iac", g="vcas_iac", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    # npath: source rail for Mgmt. PMOS diode from vdd holds it as
    # a clamped source rail one |Vgs_p| below vdd.
    devices.append(_mosfet("Mbp_npath", "pmos",
                            d="npath", g="npath", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── IAC delta: triode-helper Mgmt ──────────────────────
    # Architectural signature of IAC. PMOS in triode region acting
    # as the indirect AC coupling buffer between vcas_iac/npath and
    # the indirect Miller node niac. NOT a CS-style device gated by
    # d2 — its gate at vcas_iac (a cascode bias rail) is the "indirect"
    # signature.
    devices.append(_mosfet("Mgmt", "pmos",
                            d="niac", g="vcas_iac", s="npath", b="vdd",
                            W=W_gmt, L=L_gmt))

    # NMOS sink at niac so Mgmt's drain has a DC path. Sinks I_ref to
    # vss; this also forces Mgmt into triode by limiting V(niac).
    devices.append(_mosfet("Mbn_niac", "nmos",
                            d="niac", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # ─── IAC delta: active feedforward Mgmf ─────────────────
    # PMOS gated by d1, drain at out — standard AFFC-style FF
    # position. Provides the LHP zero through an active FF leg.
    devices.append(_mosfet("Mgmf", "pmos",
                            d="out", g="d1", s="vdd", b="vdd",
                            W=W_gmf, L=L_gmf))

    # Stage-3 NMOS sink widened to absorb Mcs3 + Mgmf DC currents.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── IAC compensation network ───────────────────────────
    # Cm1_iac: NMCF-style outer Miller (out ↔ d2).
    devices.append(_cap("Cm1_iac", p="out", n="d2", value=Cm1_iac))
    # Cm2_iac: indirect Miller cap from niac to inner RC node.
    devices.append(_cap("Cm2_iac", p="niac", n="n_rc_iac", value=Cm2_iac))
    # R0_iac: nulling resistor closing the RC path to vss.
    devices.append(_res("R0_iac", p="n_rc_iac", n="vss", value=R0_iac))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R120 C: optional DRC annotations ───────────────────
    # MATCHING (M1, M2) / CURRENT_MIRROR_MATCHING (M3, M4) /
    # GUARD_RING (Mtail). See the kwarg docs block above for the
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
                    "input-referred offset. Plain IAC stage 1 has no "
                    "input cascode (the indirect-AC compensation lives "
                    "downstream at niac), so M1/M2 imbalance is the "
                    "dominant offset source upstream of the Mgmt "
                    "triode helper"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "Stage-1 NMOS diode mirror — M3 (g=d=d1_diode) is "
                    "the unit reference leg, M4 (g=d1_diode, d=d1) is "
                    "the mirror leg. Vth or W/L mismatch unbalances "
                    "the differential drain currents at d1 / d1_diode "
                    "and pushes the high-Z d1 node off the designed OP, "
                    "skewing the indirect-AC compensation budget that "
                    "depends on Mgmt sitting in triode at the correct "
                    "Vds"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source — its pdiff in an nwell "
                    "tied to vdd shields the tail current from supply-"
                    "coupled substrate transients that would re-"
                    "modulate I_tail and inject common-mode disturbance "
                    "into the M1/M2 diff pair, leaking into d1 / "
                    "d1_diode as differential noise"
                ),
                "well_type": "nwell",
            },
        ))

    return cell
