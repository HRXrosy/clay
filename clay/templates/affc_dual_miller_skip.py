"""
Built-in template: nested_miller_3stage_affc_dual_miller_skip.

AFFC (Active FeedForward Compensation) topology overlaid with **DUAL
skip-over Miller capacitors** — a fictional-but-plausible hybrid that
combines Hoi-Lee 2003's active feedforward device gm_f2 with Fan 2005's
single-Miller-cap skip-over routing idea, except this variant uses TWO
independent skip-over caps instead of one. Each skip cap bypasses a
different two-stage segment of the 3-stage signal path.

Why this variant
================

Plain AFFC (HoiLee 2003, see `clay/templates/hoilee_affc.py`) is the
textbook NMCF baseline + active feedforward device Mgf2 (PMOS,
g=d1, d=out) + the canonical nested Miller pair (Cm1 outer between
d2 and out, Cm2 inner between d1 and d2) + passive feedforward cap Cf
(d1 to out). The active feedforward path delivers an LHP zero in the
loop transfer function at moderate frequencies; Cm1/Cm2 provide
pole-splitting.

Fan 2005 SMC (see `clay/templates/fan_smc.py`) showed that a SINGLE
skip-over Miller cap from stage-1 output directly to stage-3 output
(skipping stage 2) is sufficient for compensation in many cases — it
moves the dominant pole to the stage-1 node where the Miller
multiplier is largest.

This template is a "dual skip" generalisation:

  - Cm1_skip : cap, p=out,  n=d1    (skip cap A — d1 ↔ out, like Fan SMC)
  - Cm2_skip : cap, p=d3_in, n=d1    (skip cap B — d1 ↔ d2 stage-3-input
                                       node, bypasses stage-1-internal
                                       summing only)

Wait — but d3_in is just d2. So the second skip-over cap from d1 to
d2 IS the same routing as plain NMCF's Cm2 (inner Miller). That makes
Cm1_skip + Cm2_skip mathematically equivalent to AFFC's Cm1 + Cm2
plus a rerouting of Cm1 from "out↔d2" to "out↔d1". The fictional
holdout task is asking for this re-route: Cm1 spans stage 1 → stage 3
output (skip stage 2 entirely), Cm2 spans stage 1 → stage 2 output
(skip stage 1 input pair only). Both Miller caps now share an endpoint
at d1 (stage-1 output) which is the natural high-impedance node for
maximum Miller multiplication.

The active feedforward device Mgf2 (gate=d1, drain=out, src=vdd) is
KEPT verbatim from AFFC — that's the AFFC backbone the variant
retains. The passive Cf cap from baseline NMCF is DROPPED (the dual
skip caps already provide both feedback paths the textbook compensation
needs; Cf would over-determine the zero placement).

Topology delta vs `nested_miller_3stage_affc`
=============================================

Plain AFFC compensation block:

  Cm1 : cap, p=out, n=d2     (outer Miller, wraps stage 3)
  Cm2 : cap, p=d2,  n=d1     (inner Miller, wraps stage 2)
  Cf  : cap, p=out, n=d1     (passive feedforward, LHP zero)
  Mgf2: pmos, g=d1, d=out, s=vdd  (ACTIVE feedforward gm_f2)

AFFC + dual Miller skip compensation block:

  Cm1_skip : cap, p=out, n=d1   (skip cap A — d1 ↔ out, NEW routing)
  Cm2_skip : cap, p=d2,  n=d1   (skip cap B — d1 ↔ d2, same as plain
                                  AFFC's Cm2 but renamed for cohesion)
  Mgf2     : pmos, g=d1, d=out, s=vdd  (KEPT verbatim from AFFC)

The architectural signature is the pair of caps named `Cm1_skip` and
`Cm2_skip` (no other Clay stdlib variant has BOTH skip-over caps
sharing the d1 endpoint), plus the AFFC `Mgf2` device. The instance
names `Cm1_skip` / `Cm2_skip` (not Cm1/Cm2/Cf) are the disambiguating
markers `tests/test_fictional_combo_stdlib_v92.py` asserts.

Why share the d1 endpoint
=========================

In a 3-stage opamp, d1 (stage-1 output) is the natural high-Z node
for Miller compensation: stage-1 is loaded by the stage-2 input
capacitance Cgs(Mcs2) plus parasitic gate-drain caps and (when
Miller compensation is present) the Miller-multiplied feedback cap.
The Miller-multiplier on a cap from d1 to a downstream node N is
roughly the gain from d1 to N — large if N is past stage 2 (out has
gain A2·A3 vs d1), smaller if N is just d2 (only A2 of multiplication).

Routing Cm1_skip from d1 directly to out lets that cap inherit the
maximum Miller multiplication (A2·A3) → pushes the dominant pole at
d1 to the lowest frequency for a given physical capacitance. Routing
Cm2_skip from d1 to d2 only gives A2 of multiplication — useful for
trimming the second pole's location independently.

Combined with the active Mgf2 path (bypasses stage 2 entirely with a
real gm), the loop transfer function gets:
  - 1 dominant pole at d1 (loaded by Cm1_skip · A2·A3 + Cm2_skip · A2)
  - 1 LHP zero from Mgf2's active feedforward path
  - 1 additional LHP-or-RHP zero from Cm1_skip's direct feedback (sign
    depends on gm_3 vs Cm1_skip ratio).

The dual-cap arrangement gives the designer ONE degree of freedom more
than Fan's single-cap SMC (Cm2_skip lets you trim the second pole's
location without touching the dominant-pole capacitance) and ONE
degree of freedom less than AFFC + Cf (the explicit feedforward cap
Cf is dropped).

Bias plan
---------

Identical to the plain AFFC baseline — including the stage-3 NMOS
sink scale-up (Mlcs3 widened by `(1 + Mgf2_scale)` to absorb both
Mcs3 and Mgf2 DC currents). See `clay/templates/hoilee_affc.py` for
the full bias-chain documentation.

Knobs
-----

The template exposes the AFFC knobs PLUS two skip-over cap values:

  Cm1_skip: float = 1.5e-12
      Skip cap A — d1 ↔ out. Default matches plain AFFC's Cm1 default.
  Cm2_skip: float = 0.5e-12
      Skip cap B — d1 ↔ d2 (same routing as plain AFFC's Cm2). Default
      matches plain AFFC's Cm2 default.

AFFC knobs inherited verbatim:

  Mgf2_scale: float = 1.0      — ratio of Mgf2 W to Mcs3 W
  W_gf2, L_gf2: optional explicit overrides
  gm_id_gf2: gm/Id intent override for Mgf2

The `Cf` kwarg from the plain AFFC baseline is NOT exposed because
this variant drops Cf entirely. The kwargs `Cm1` and `Cm2` from the
plain AFFC baseline are NOT exposed — the routing is structurally
different so the cap names are different.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load,
W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3,
gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3) are inherited
verbatim from the plain AFFC baseline.
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


def build_nested_miller_3stage_affc_dual_miller_skip(
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

    # ─── DUAL-SKIP compensation: TWO skip-over Miller caps ──
    # Both caps share an endpoint at d1 (stage-1 output). Cm1_skip
    # bypasses BOTH stage 2 and stage 3 (skips over the entire output
    # path), Cm2_skip only bypasses stage 2 (same routing as plain
    # AFFC's Cm2 but renamed to disambiguate the cohort).
    Cm1_skip: float = 1.5e-12,
    Cm2_skip: float = 0.5e-12,

    # ─── AFFC-specific: active feedforward (gm_f2) ──────────
    Mgf2_scale: float = 1.0,
    W_gf2: float | None = None,
    L_gf2: float | None = None,
    gm_id_gf2: float | None = None,

    # ─── R116 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # AFFC dual-skip 3-stage opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset; the
    #     dual-skip routing pins d1 (M2 drain) as the dominant-pole node,
    #     so any DC offset at d1 propagates through both Miller caps and
    #     is amplified by stages 2 and 3 to the output.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the high-Z d1 node (the dual-skip dominant-pole
    #     node carrying both Cm1_skip and Cm2_skip terminations) from
    #     supply-coupled substrate noise.
    #   * PASSIVE_MATCHING on the dual skip-over Miller cap pair
    #     (Cm1_skip / Cm2_skip). Both caps share the d1 endpoint; their
    #     ratio sets the relative weight of the out→d1 vs d2→d1 skip
    #     paths. The fringing-cap unit-array primitive preserves the
    #     ratio against process gradients better than independently
    #     sized caps.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for an AFFC opamp with dual skip-over Miller caps.

    Same 5-port surface as plain AFFC. The dual-skip routing reorders
    the cap endpoints so both Miller caps share d1 (stage-1 output),
    making it the dominant-pole node; the active gm_f2 device is
    retained verbatim from the AFFC baseline. The passive Cf cap from
    NMCF is dropped.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see AFFC / NMCF baseline docstrings).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline signal path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four baseline signal devices.
        Cm1_skip: skip cap A (d1 ↔ out). Default 1.5 pF.
        Cm2_skip: skip cap B (d1 ↔ d2). Default 0.5 pF.
        Mgf2_scale: ratio of Mgf2 W to Mcs3 W (default 1.0).
        W_gf2, L_gf2: optional explicit overrides for the feedforward
            device sizing.
        gm_id_gf2: gm/Id intent override for Mgf2. When set the
            closed-form solver derives W_gf2 from (gm_id, I_d, L) where
            I_d is `Mgf2_scale * K_cs3_load * I_ref`.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF/AFFC).
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs2 = I_diff
    I_cs3 = K_cs3_load * I_ref

    # AFFC: Mgf2 draws additional DC current proportional to Mgf2_scale.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref

    # NMOS sink absorbs Mcs3 + Mgf2 DC currents — bump mirror ratio.
    K_cs3_load_total = K_cs3_load * (1 + Mgf2_scale)

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

    # Mgf2 sizing — same priority as plain AFFC (gm/Id beats explicit
    # W_gf2 beats Mgf2_scale-derived default).
    if L_gf2 is None:
        L_gf2 = L_cs3
    if gm_id_gf2 is not None:
        W_gf2 = closed_form_W("pmos", gm_id=gm_id_gf2,
                              I_d=I_gf2, L=L_gf2).W
    elif W_gf2 is None:
        W_gf2 = W_cs3 * Mgf2_scale

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

    # ─── Stage 2 replica-bias leg (unchanged from AFFC) ──────
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

    # ─── AFFC: active feedforward device gm_f2 (KEPT verbatim) ──
    # PMOS in parallel with Mcs3 but gated by d1 (stage-1 output)
    # instead of d2 (stage-2 output). Bypasses stage 2 entirely.
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                           W=W_gf2, L=L_gf2))

    # Stage-3 NMOS sink (mirror of nbias) — widened to absorb both
    # Mcs3 and Mgf2 DC currents (same as plain AFFC).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                           W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: DUAL SKIP-OVER Miller caps ────────────
    # Architectural signature: BOTH caps share an endpoint at d1
    # (stage-1 output). Cm1_skip routes from d1 directly to out
    # (skipping stage 2 entirely — like Fan SMC's single Miller cap).
    # Cm2_skip routes from d1 to d2 (same routing as plain AFFC's
    # Cm2 but renamed to disambiguate the dual-skip cohort).
    # No Cm1 (out↔d2 routing of plain AFFC), no Cf (passive
    # feedforward dropped — the active Mgf2 path subsumes it).
    devices.append(_cap("Cm1_skip", p="out", n="d1", value=Cm1_skip))
    devices.append(_cap("Cm2_skip", p="d2",  n="d1", value=Cm2_skip))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R116 C: optional DRC annotations ─────────────────
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
                    "input-referred offset; common-centroid required. "
                    "Dual-skip routing pins d1 (M2 drain) as the "
                    "dominant-pole node carrying both Miller cap "
                    "terminations, so M1/M2 mismatch is amplified by "
                    "stages 2 and 3 to the output"
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
                    "to vdd shields the high-Z d1 node (the dual-skip "
                    "dominant-pole node carrying both Cm1_skip and "
                    "Cm2_skip terminations) from supply-coupled "
                    "substrate noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1_skip", "Cm2_skip"),
            extras={
                "rationale": (
                    "Dual skip-over Miller cap pair — both terminate at "
                    "d1; their ratio sets the relative weight of the "
                    "out↔d1 vs d2↔d1 skip paths. Fringing-cap unit-array "
                    "primitive preserves the ratio against process "
                    "gradients better than independently sized caps"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
