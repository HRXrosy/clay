"""
Built-in template: nested_miller_3stage_nmcf.

Three-stage opamp with **nested Miller compensation only** — the
canonical Leung-Mok NMCF baseline as published, WITHOUT the
feedforward cap Cf. This is the textbook NMCF (Nested Miller
Compensation) starting point against which feedforward (Cf) and
active-feedforward (Mgf2) variants are typically compared.

Why a separate template (vs the existing `_feedforward` variant)
================================================================

The existing `nested_miller_3stage_feedforward` template embeds Cf
(d1 ↔ out) by default — it materialises NMCF + passive feedforward.
For combinator algebra and pedagogical "what does NMCF look like
without any feedforward zero?" questions, a clean baseline that
contains ONLY the two nested Miller caps Cm1 (outer) and Cm2 (inner)
is the right starting point. Removing the feedforward cap lets the
combinator layer add `feedforward(stage_a, stage_c, gain=...)` as
an explicit composition rather than baking it into the base cell.

Topology
========

Signal path: identical to `nested_miller_3stage_feedforward`:

  - Stage 1: PMOS diff pair (M1/M2) + NMOS mirror load (M3/M4).
  - Stage 2: NMOS CS (Mcs2) with replica-bias PMOS load (Mlcs2 +
    Mcs2_repl + Mbp_d2).
  - Stage 3: PMOS CS (Mcs3) + NMOS sink (Mlcs3).
  - Bias chain: Mbp + Mbn + Mbnm + Mtail + Iref.

Compensation network (deliberately minimal):

  Cm1 : OUTER nested Miller, between d2 and out (wraps stage 3).
  Cm2 : INNER nested Miller, between d1 and d2 (wraps stage 2).
  (NO Cf — that is the whole point of this "baseline" variant.)

Without Cf, the textbook NMCF stage has its RHP zero from the
outer Miller path (Cm1 routing the stage-3 input forward through
Mcs3) un-cancelled. The closed-loop phase margin is therefore
lower than the `_feedforward` variant at the same Cm1/Cm2 values.
In practice users adding gain margin will either:

  (a) Add an explicit `feedforward(stage_a, stage_c)` combinator
      that injects a LHP zero, OR
  (b) Compose `compensate(stage_c, zero=Rz1)` to add a nulling-R,
      OR
  (c) Increase Cm1 so the dominant pole drops below crossover.

This baseline forces those design choices to be explicit.

Same 5-port surface as the `_feedforward` variant
(`inn`, `inp`, `out`, `vdd`, `vss`). Reuses the same closed-form
gm/Id sizing path. The only kwargs delta is the absence of Cf.
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


def build_nested_miller_3stage_nmcf(
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

    # ─── Compensation caps (NMCF baseline — Cm1 + Cm2 only) ─
    Cm1: float = 2.0e-12,   # outer nested Miller (d2 ↔ out)
    Cm2: float = 0.7e-12,   # inner nested Miller (d1 ↔ d2)

    # ─── R112 C: optional qualitative DRC annotations ───────
    # When True, attach a CURRENT_MIRROR_MATCHING annotation on the
    # stage-1 NMOS load mirror group (M3 unit / M4 mirror leg /
    # Mcs2_repl stage-2 replica). All three share gate `d1_diode` at
    # identical W/L, forming a unit-cell array that the layout prompt
    # MUST realise as interdigitated unit cells (not W-multiplier-only)
    # so the ratio is preserved across local-mismatch and OD-edge
    # gradients. The new R111 C CURRENT_MIRROR_MATCHING kind carries
    # the unit-device hint explicitly so the layout prompt knows which
    # leg is the reference. Annotations live only in
    # cell.__dict__["_drc_annotations"]; SPICE emit (which iterates
    # cell.devices) is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the NMCF baseline 3-stage opamp on sky130.

    Same 5-port surface as the `_feedforward` variant; the only delta
    is no Cf (feedforward cap absent). Default Cm1/Cm2 are slightly
    bumped over the `_feedforward` defaults to compensate for the
    missing LHP zero — without Cf the inner pole-splitting must do
    more work to keep the phase margin above 45 deg at moderate loads.

    kwargs are identical to `_feedforward` except `Cf` is not accepted.
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as _feedforward).
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

    # ─── Bias chain ──────────────────────────────────────────
    devices.append(_mosfet("Mbp",  "pmos", d="pbias", g="pbias",
                            s="vdd", b="vdd", W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn",  "nmos", d="nbias", g="nbias",
                            s="vss", b="vss", W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbnm", "nmos", d="pbias", g="nbias",
                            s="vss", b="vss", W=W_bn, L=L_bn))
    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    # ─── Stage 1: PMOS diff pair with NMOS mirror load ───────
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias",
                            s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos", d="d1_diode", g="inp",
                            s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1", g="inn",
                            s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1",
                            s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg ────────────────────────────
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode",
                            s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2",
                            s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2",
                            s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── Stage 3: PMOS CS amplifier (output stage) ───────────
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2",
                            s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias",
                            s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── Compensation: nested Miller caps ONLY ───────────────
    # Cm1 OUTER (wraps stage 3): d2 ↔ out.
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    # Cm2 INNER (wraps stage 2): d1 ↔ d2.
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))
    # NO Cf — that is the architectural marker of the NMCF baseline
    # vs the _feedforward variant.

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R112 C: optional CURRENT_MIRROR_MATCHING annotation ─
    # The stage-1 NMOS load mirror M3 (diode) and M4 (mirror leg) sit
    # at d1_diode and d1 respectively; Mcs2_repl is a third leg at the
    # same gate (d1_diode) that replicates Mcs2's DC current for the
    # stage-2 replica-bias scheme. All three share W/L=W_load1/L_load1
    # at the NMCF baseline default (or the gm/Id solver's
    # gm_id_load1 / gm_id_stage2 outputs at intent form). The R111 C
    # CURRENT_MIRROR_MATCHING kind says "lay these out as a unit-cell
    # array, not as a W-multiplier" — required because the matching
    # of M3↔M4 sets the stage-1 mirror error AND the matching of
    # M3↔Mcs2_repl sets the stage-2 replica's bias accuracy.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4", "Mcs2_repl"),
            extras={
                "rationale": (
                    "NMCF stage-1 NMOS load mirror with stage-2 replica leg — "
                    "all three legs share gate d1_diode and W/L; layout MUST "
                    "use interdigitated unit cells so M4 mirrors M3 and "
                    "Mcs2_repl replicates M3's bias point"
                ),
                "ratio": "1:1:1",
                "unit_device": "M3",
                "primitive": "interdigitated_array",
            },
        ))

    return cell
