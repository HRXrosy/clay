"""
Built-in template: nested_miller_3stage_smc.

Three-stage opamp with **Single Miller Compensation** (SMC) — the
minimalist Fan 2005 compensation scheme that uses ONE Miller cap
spanning stage-1 output to stage-3 output (skip-over compensation),
rather than the textbook Nested Miller scheme's two caps Cm1 + Cm2.

Reference
=========

Fan, X., Kakkar, K., and Karki, J., "Single Miller Capacitor
Frequency Compensation for Three-Stage Amplifiers," ISCAS 2005
(also analysed in JSSC 2005). The headline insight: most of the
parasitic damage in textbook NMCF comes from the inner Miller cap
Cm2 (between stage 2 and stage 3 outputs), which forces a RHP zero
that the feedforward cap Cf is then needed to cancel. Drop Cm2 and
route Cm1 directly from stage-1 output to stage-3 output and the
RHP zero never forms — you only need ONE cap.

Topology delta vs `nested_miller_3stage`
========================================

The signal path inherits the NMCF baseline verbatim (PMOS diff
pair → NMOS-mirror load → NMOS CS stage 2 → PMOS CS stage 3 with
NMOS sink), including the stage-2 replica-bias leg (Mcs2_repl +
Mbp_d2 + Mlcs2). Only the compensation network changes.

Compensation device deltas
--------------------------

NMCF baseline has three compensation devices:

  Cm1 : cap, p=out, n=d2     (outer Miller, wraps stage 3)
  Cm2 : cap, p=d2,  n=d1     (inner Miller, wraps stage 2)
  Cf  : cap, p=out, n=d1     (passive feedforward, LHP zero)

SMC keeps only ONE Miller cap, re-routed:

  Cm1 : cap, p=out, n=d1     (skip-over: spans stage 1 → stage 3)

Cm2 is **removed entirely**. Cf is also removed — the skip-over
routing of Cm1 already provides the dominant feedback path Fan
2005 analyses, and the textbook Cf was specifically there to
cancel the RHP zero introduced by Cm2 in the NMCF scheme. With
Cm2 gone, the RHP zero never forms and Cf has nothing to cancel.

Schematic (compensation network delta only — signal path unchanged)
-------------------------------------------------------------------

  NMCF:                              SMC (Fan 2005):
    d1 ─── Cm2 ─── d2                  d1 ─── Cm1 ─── out
    d2 ─── Cm1 ─── out                 (single cap, skip-over)
    d1 ─── Cf ──── out                 (no Cm2, no Cf)

Why one cap suffices (Fan 2005 transfer-function argument)
----------------------------------------------------------

The loop gain of a 3-stage amplifier has 3 high-impedance nodes
(d1, d2, out) and therefore 3 poles. NMCF uses pole-splitting on
Cm1 and Cm2 to move two of those poles to high frequency while
keeping a dominant pole at d2 (loaded by Cm1). But Cm2's pole-
splitting creates a parasitic RHP zero at s = +gm_3 / Cm2 (the
classic Miller RHP zero, just at the inner stage).

Fan's argument: skip Cm2 entirely. Route Cm1 from d1 directly to
out and the dominant pole still sits at the high-Z node d1 (now
loaded by Cm1 multiplied by gain stages 2 AND 3 — bigger Miller
multiplication). The pole at d2 stays at its natural frequency
(1 / (2π · r_o2 · C_d2)) but is pushed to higher frequency by
the relatively small parasitic C_d2 (no Miller cap loading d2 →
just gate caps). The pole at out is pushed even higher by direct
loading from gm_3 (Mcs3 transconductance) into a small C_load.

The price Fan pays: only ONE compensation cap means the zero
placement is harder to control. The single Miller cap creates a
zero at s = +gm_3 / Cm1 — which is RHP for default sizing. Fan's
fix is to require gm_2 >= gm_3 (the inner stage's transconductance
should exceed the output stage's). Under that constraint, the
zero migrates to LHP through a different mechanism — the inner
stage's feedforward of the input signal through Mcs2's gate-drain
parasitic combined with the small loop gain at d2 effectively
cancels the RHP zero seen at the output. In practice this means
**stage 2 must be sized with a generous gm/Id** so that gm_2 is
large enough vs gm_3.

The Clay implementation honors this by exposing a `gm2_boost`
kwarg (default 1.5) that internally widens W_cs2 by this factor
when used with `gm_id_stage2` or `W_cs2` — pushing gm_2 above
gm_3 to keep the zero in LHP per Fan's recommendation. Set
`gm2_boost = 1.0` to disable the boost and recover the same sizing
as plain NMCF for direct comparison.

Reference netlist (AnalogGym `Amplifier/spice_netlist/Fan_SMC_Pin_3`)
--------------------------------------------------------------------

AnalogGym's reference Fan_SMC netlist has exactly one Miller cap:

  Cm1  net_out  net_d1   <value>   (single skip-over cap)

That cap spans the output and a stage-1-output equivalent net. The
Clay analogue is `Cm1` with p=out, n=d1 — exactly the topology
this template materialises.

What you should NOT expect from SMC (honest scope)
--------------------------------------------------

SMC is a **simpler** compensation scheme, not a better one. The
v0.0.51 PVT envelope on gf180mcu shows (with the default sizing):

  - DC gain: roughly similar magnitude to NMCF baseline (the
    signal path is identical, the comp caps only affect AC). Small
    delta (±2 dB) due to load-pole shift at d2 when Cm2 is removed.
  - GBW: changes substantially because the dominant pole is at a
    different node (d1 loaded by Miller-multiplied Cm1) vs NMCF's
    (d2 loaded by Cm1). Direction of change depends on the ratio
    of stage 2 and stage 3 gains and Cm1 value — typically GBW is
    in the same ballpark as NMCF or slightly lower.
  - Phase margin: depends on gm2/gm3 ratio. The default
    `gm2_boost = 1.5` is conservative; Fan 2005 uses up to 2× for
    aggressive cases.

The distinguishing test is the materialised SPICE netlist: SMC
has exactly ONE Miller cap (Cm1, p=out, n=d1), no Cm2, no Cf;
NMCF has THREE compensation caps (Cm1, Cm2, Cf).
`tests/test_fan_smc_distinct.py` asserts this invariant.

Knobs
-----

The template exposes one new kwarg on top of the NMCF baseline:

  gm2_boost: float = 1.5
      Multiplier on W_cs2 (when gm_id_stage2 is used) or directly
      on the W_cs2 kwarg (when W_cs2 is set explicitly). Default
      1.5 widens the stage-2 NMOS CS device by 50 % vs plain NMCF,
      pushing gm_2 above gm_3 to keep the single Miller cap's
      zero in LHP per Fan 2005. Set to 1.0 to disable the boost.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3) are inherited
verbatim from the NMCF baseline. The kwargs Cm2 and Cf from the
NMCF baseline are **not exposed** because the SMC topology does
not contain those devices — the only Miller cap value is `Cm1`.
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


def build_nested_miller_3stage_smc(
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

    # ─── SMC compensation: ONE Miller cap ───────────────────
    Cm1: float = 1.5e-12,   # single skip-over Miller (d1 ↔ out)

    # ─── SMC-specific: gm2 boost to keep zero in LHP ────────
    # Fan 2005 requires gm_2 > gm_3 for the single Miller cap's
    # zero to migrate to LHP. The default 1.5 widens W_cs2 by 50 %
    # vs plain NMCF, which (at the same gm/Id) gives gm_2 ≈ 1.5×
    # the baseline stage-2 transconductance. Set to 1.0 to disable.
    gm2_boost: float = 1.5,

    # ─── R122 C: qualitative DRC annotation surface ────────
    # Additive metadata. Default False keeps every pre-R122 caller
    # byte-for-byte identical. When True, three qualitative hints
    # attach to the cell:
    #   MATCHING                on (M1, M2)     -- PMOS diff pair
    #   CURRENT_MIRROR_MATCHING on (M3, M4)     -- NMOS diode mirror
    #   GUARD_RING              on (Mtail,)     -- PMOS tail (nwell)
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a Fan 2005 SMC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells and the NMCF
    baseline. SMC differs from NMCF only in the compensation network
    (ONE Miller cap spanning d1 ↔ out instead of three caps).

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1: single skip-over Miller cap value (d1 ↔ out). Default
            1.5 pF matches the NMCF baseline outer Miller cap.
        gm2_boost: multiplier on W_cs2 to push gm_2 above gm_3 per
            Fan 2005. Default 1.5. Set to 1.0 to disable.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0     # each PMOS diff-pair leg
    I_load1 = I_diff                    # each NMOS mirror leg
    I_cs2 = I_diff                      # ≈ I_M3 stage-2 NMOS CS current
    I_cs3 = K_cs3_load * I_ref          # stage-3 PMOS CS current

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_load1, L=L_load1).W
    if gm_id_stage2 is not None:
        # Apply gm2_boost after the closed-form solver derives W. This
        # keeps the gm/Id intent at the user-declared value (the device
        # still operates at the requested gm/Id) but scales physical W
        # to multiply the resulting gm. Net effect: gm_2 increases by
        # gm2_boost × at fixed gm/Id and fixed I_d.
        W_cs2 = closed_form_W("nmos", gm_id=gm_id_stage2,
                              I_d=I_cs2, L=L_cs2).W * gm2_boost
    else:
        # When the user passed a physical W_cs2, apply gm2_boost
        # directly so the SMC default still widens stage 2 vs an NMCF
        # cell with identical W kwargs.
        W_cs2 = W_cs2 * gm2_boost
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
    # Mcs2: NMOS, gate=d1, drain=d2, source=vss. Widened by gm2_boost
    # so gm_2 exceeds gm_3 per Fan 2005's LHP-zero constraint.
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg ────────────────────────────
    # Same replica-bias scheme as NMCF baseline. The replica device
    # is sized identically to Mcs2 (already boosted by gm2_boost) so
    # the DC current it sinks matches Mcs2's at its real OP.
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

    # Stage-3 NMOS sink (mirror of nbias). SMC does NOT add an active
    # feedforward device so K_cs3_load is used as-is (no multiplier).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── Compensation: ONE Miller cap (Fan 2005 SMC) ─────────
    # The single Miller cap is the architectural signature of Fan 2005
    # SMC: spans d1 (stage-1 output) directly to out (stage-3 output),
    # skipping over stage 2. No Cm2, no Cf. The textbook NMCF emits
    # THREE caps (Cm1, Cm2, Cf); SMC emits exactly one (Cm1 only).
    devices.append(_cap("Cm1", p="out", n="d1", value=Cm1))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R122 C: optional qualitative DRC annotations ─────────
    # Three annotations attached only when attach_drc=True; default
    # path is byte-identical to pre-R122.
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
                    "input-referred offset. SMC's single skip-over "
                    "Miller cap (d1<->out) amplifies stage-1 d1-node "
                    "offset through both stages 2 and 3 to ``out``; "
                    "matching the diff pair is the prerequisite for "
                    "Fan 2005's gm_2 > gm_3 LHP-zero migration to "
                    "operate on a clean DC operating point"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "NMOS stage-1 diode mirror (M3 diode, M4 mirror "
                    "leg). Mirror ratio error injects a static "
                    "differential current into d1 — the very node the "
                    "single skip-over Miller cap couples to ``out``, "
                    "so any 1:1 mismatch maps directly to output DC "
                    "offset through Cm1's gain path; common-centroid "
                    "unit-cell array required to hold the ratio"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) sinks the "
                    "full stage-1 bias current from vdd into tail. "
                    "An nwell guard ring tied to vdd shields the "
                    "high-Z d1 node from supply-coupled substrate "
                    "noise that would otherwise modulate the "
                    "skip-over Miller path and degrade SMC's "
                    "single-cap PSRR margin"
                ),
                "well_type": "nwell",
            },
        ))

    return cell
