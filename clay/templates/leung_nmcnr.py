"""
Built-in template: nested_miller_3stage_nmcnr.

Three-stage opamp with **Nested Miller compensation with Nulling
Resistor** (NMCNR) — Leung's variant of the NMCF family that adds an
explicit nulling resistor in series with the outer Miller cap so that
the RHP zero from the Miller feedforward path can be moved into the
left-half plane (or to infinity) by sizing R · gm_3 ≈ 1 / gm_3.

Reference
=========

The Leung_NMCNR_Pin_3 reference netlist in AnalogGym
(`Amplifier/spice_netlist/Leung_NMCNR_Pin_3`) implements this
topology with a single Miller cap plus a series resistor R0 +
a second compensation cap C1 at the inner pole node. The headline
insight: textbook NMCF's outer Miller cap creates a RHP zero at
s = +gm_3 / Cm1 that the feedforward cap Cf must cancel. NMCNR
replaces Cf with a series resistor R inside the Miller cap branch,
so the zero is now at s = (1 - gm_3 · R) / (R · Cm1) — pick
R ≈ 1 / gm_3 and the zero moves to infinity, or pick R > 1 / gm_3
and it migrates to LHP.

The architectural signature in the reference netlist:
  C0  net050  net044            (outer Miller cap to nulling-R node)
  R0  net044  VOUT              (nulling resistor, in series with C0)
  C1  net044  net049            (inner Miller cap, also through nulling-R node)

In the Clay implementation we mirror this with two caps Cm1 + Cm2
sharing a common nulling-R node `rz_n` between the d2/d1 inner side
and the output stage:

  Cm1 : cap, p=rz_n, n=d2      (outer Miller, terminates at nulling-R node)
  Rz1 : res, p=rz_n, n=out     (nulling resistor between rz_n and out)
  Cm2 : cap, p=rz_n, n=d1      (inner Miller, also through nulling-R node)

The `rz_n` node is what makes this template distinct: it's a real
internal node that doesn't exist in NMCF, SMC, CFCC, or AFFC.

Topology delta vs `nested_miller_3stage`
========================================

Stage 1, stage 2, and stage 3 (signal path) are inherited verbatim
from the NMCF baseline. Only the compensation network changes:

NMCF baseline compensation:
  Cm1 : cap, p=out, n=d2     (outer Miller, wraps stage 3 directly)
  Cm2 : cap, p=d2,  n=d1     (inner Miller, wraps stage 2)
  Cf  : cap, p=out, n=d1     (passive feedforward LHP-zero generator)

NMCNR compensation (Leung):
  Cm1 : cap, p=rz_n, n=d2    (outer Miller, terminates at the nulling-R node)
  Rz1 : res, p=rz_n, n=out   (nulling resistor)
  Cm2 : cap, p=rz_n, n=d1    (inner Miller, ALSO routed through rz_n)

The feedforward cap Cf is dropped — the nulling-R technique handles
RHP-zero cancellation without needing Cf. The architectural signature
is the resistor `Rz1` (no other 3-stage variant in the Clay stdlib
has one), and the shared `rz_n` node connecting both Miller caps to
the nulling resistor.

NB: AnalogGym's reference Leung_NMCNR_Pin_3 emits an explicit
resistor primitive (R0 in their netlist). The Clay IR maps this to
`device_class="res"` with `pins={"p","n"}` and `sizing={"value": ...}`.

Why nulling-R compensation matters (transfer-function level)
------------------------------------------------------------

The textbook Miller cap creates a RHP zero at s = +gm_3 / Cm1. In a
3-stage amp this RHP zero shows up at frequencies comparable to GBW
and reduces phase margin — sometimes catastrophically. Cf in plain
NMCF is one fix (creates an LHP zero that cancels the RHP); the
nulling-R is another (modifies the Miller-cap zero directly).

With Rz1 in series with Cm1, the zero migrates to:

  s_z = 1 / (Cm1 · (1/gm_3 - Rz1))

Pick Rz1 = 1 / gm_3 → zero at infinity (cancellation).
Pick Rz1 > 1 / gm_3 → zero in LHP (helps PM further).
Pick Rz1 < 1 / gm_3 → zero in RHP but at higher frequency than the
                       textbook s = gm_3 / Cm1.

Default sizing in this template: Rz1 = 5 kΩ (chosen so that with
default gm_3 around 200 µS, Rz1 · gm_3 ≈ 1, putting the zero near
infinity). Cm1 stays at 1.5 pF; Cm2 stays at 0.5 pF.

Schematic (compensation network delta — signal path unchanged)
---------------------------------------------------------------

  NMCF:                              NMCNR (Leung):
    d1 ── Cm2 ── d2                    d1 ── Cm2 ── rz_n ── Rz1 ── out
    d2 ── Cm1 ── out                   d2 ── Cm1 ──┘
    d1 ── Cf ─── out                   (no Cf, but Rz1 sits in series
                                        with the Miller path)

Honest scope (what you should NOT expect)
-----------------------------------------

NMCNR is a textbook compensation refinement; the gain remains at
NMCF baseline (signal path unchanged). Tuning Rz1 vs gm_3 gives you
a degree of freedom to trade GBW for PM, but the default sizing
above is conservative — expect numbers in the same ballpark as
plain NMCF (gain ≈ 58 dB / GBW ≈ 870 kHz on gf180mcu tt/27 °C).

The distinguishing test is the materialised IR: NMCNR contains a
device of class `res` named `Rz1`, and Cm1 endpoints are
{rz_n, d2} (not {out, d2} like NMCF). Tests in
`tests/test_leung_nmcnr_distinct.py` assert these invariants.

Knobs
-----

NMCNR adds one new kwarg on top of the NMCF baseline:

  Rz1: float = 5e3
      Nulling resistance in series with Cm1 (between rz_n and out).
      Setting Rz1 ≈ 1/gm_3 moves the RHP zero to infinity; values
      above this push it to LHP, values below leave it RHP at
      higher frequency.

The kwarg `Cf` from the NMCF baseline is NOT exposed — NMCNR
explicitly drops Cf in favor of the nulling-R mechanism.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3, Cm1, Cm2) are
inherited verbatim from the NMCF baseline.
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


def build_nested_miller_3stage_nmcnr(
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

    # ─── NMCNR compensation: TWO Miller caps + nulling R ────
    Cm1: float = 1.5e-12,   # outer Miller (rz_n ↔ d2)
    Cm2: float = 0.5e-12,   # inner Miller (rz_n ↔ d1)
    Rz1: float = 5e3,       # nulling resistor (rz_n ↔ out)

    # ─── R115 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # Leung NMCNR 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node from supply-coupled
    #     substrate noise (especially important because NMCNR routes
    #     both Cm1 and Cm2 through a shared rz_n node which is sensitive
    #     to common-mode supply coupling).
    #   * PASSIVE_MATCHING on the Miller cap pair (Cm1 / Cm2). Hastings
    #     Ch. 8 — the NMCNR compensation network's pole-zero placement
    #     depends on the Cm1 / Cm2 ratio (outer vs inner Miller cap),
    #     not just absolute values. Both caps terminate at the shared
    #     rz_n node, so any local mismatch shifts the doublet location
    #     and the nulling resistor's RHP-zero-cancellation alignment.
    #     Fringing-cap unit array required so the ratio survives
    #     local-mismatch gradients.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Leung NMCNR 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells, the NMCF baseline,
    and the SMC / AFFC / CFCC variants. NMCNR differs from NMCF only in
    the compensation network — a nulling resistor `Rz1` is inserted
    between the Miller-cap node `rz_n` and the output. Both Miller caps
    Cm1 (outer) and Cm2 (inner) terminate at `rz_n` instead of the
    NMCF baseline's out/d2/d1 routings.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1: outer Miller cap (rz_n ↔ d2). Default 1.5 pF.
        Cm2: inner Miller cap (rz_n ↔ d1). Default 0.5 pF.
        Rz1: nulling resistance in series with Cm1 (rz_n ↔ out).
            Default 5 kΩ. Pick ≈ 1/gm_3 to cancel the RHP zero.

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

    # ─── Compensation: NMCNR (nulling resistor + 2 Miller caps) ─
    # Architectural signature: the `rz_n` shared node + the Rz1
    # series resistor. Both Miller caps Cm1 (outer) and Cm2 (inner)
    # terminate at rz_n; Rz1 connects rz_n to the output.
    # AnalogGym's reference Leung_NMCNR_Pin_3 has this exact shape
    # (C0/C1 sharing net044, R0 from net044 to VOUT). NMCF baseline
    # routes Cm1 directly to out and adds a separate Cf cap; the
    # rz_n / Rz1 introduction is the NMCNR distinction.
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

    # ─── R115 C: optional DRC annotations ─────────────────
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
                    "entire stage-1 bias current; nwell guard ring tied "
                    "to vdd shields the high-Z d1 node from supply-"
                    "coupled substrate noise (NMCNR routes both Cm1 / "
                    "Cm2 through a shared rz_n node sensitive to common-"
                    "mode supply coupling)"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "NMCNR Miller cap pair (outer + inner) — pole-zero "
                    "placement depends on the Cm1 / Cm2 ratio not "
                    "absolute values; both caps terminate at the shared "
                    "rz_n node so local mismatch shifts doublet location "
                    "and nulling-resistor RHP-zero alignment; fringing-"
                    "cap unit array required to preserve ratio across "
                    "local-mismatch gradients"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
