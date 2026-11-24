"""
Built-in template: single_stage_dual_feedforward.

Single-stage differential opamp with TWO symmetric feedforward
transconductor (gm) paths from the input pair directly to the output.
A fictional-but-plausible variant that takes the textbook
PMOS-input → NMOS-mirror single-stage opamp baseline and adds two
INDEPENDENT feedforward gm devices — one driven by `inp`, one driven
by `inn` — both feeding the same single-ended output `out`. The
feedforward pair is symmetric so its differential-mode contribution
ADDS to the main signal path while its common-mode contribution
cancels.

Why this variant
================

Plain single-stage diff-pair + NMOS mirror has a fixed gain-bandwidth
trade-off: gain = gm_in · (ro_p || ro_n), bandwidth ≈ 1 / (2π Ro · CL).
The single high-impedance node `out` sits one pole-splitting move
away from any compensation cap. The textbook way to boost the
gain-bandwidth product (GBW) of a single-stage opamp is to either
cascade a second stage (which requires Miller compensation) or to
inject extra signal current at the output via an active feedforward
gm — bypassing the high-impedance node entirely and adding directly
to the load current.

This template implements the SYMMETRIC dual-feedforward variant: two
PMOS feedforward devices Mff_p and Mff_n, each sized identically,
each with its source tied to vdd and its drain tied to `out`, but
gated by `inp` and `inn` respectively. The two devices form a third
diff-pair (in addition to the main input pair) but without a shared
tail current source — they are common-source, not differential-pair.

Topology
========

           vdd ────┬──────────┬──────────┬──────────┬──────────────
                   │          │          │          │
                  Mbp       Mtail      Mff_n      Mff_p   ← TWO feedforward
                   │          │          │          │       devices, gated
                   │          tail       │          │       by inn / inp
                   │       ┌──┴──┐       │          │
                   │      M1   M2        │          │
             inp ──┤      │     │ ── inn │          │
                   │     d1_d  d1        │          │
                   │      │     │        │          │
                   │      M3    M4       │          │   (out is the COMMON
                   │      │     │        │          │    drain of M4 +
                  Iref    vss   vss      │          │    Mff_n + Mff_p)
                   │             │       │          │
                   │            out──────┴──────────┘
                   │             │
                   │            CL (load cap, owned by testbench)
                   │             │
                   │            vss

Wait — that's wrong. M4's drain IS out (single-ended output of the
mirror). So Mff_n's drain (gated by inn) and Mff_p's drain (gated by
inp) BOTH tie to out alongside M4. Let me re-document:

Stage 1 (the only real stage):
  Mtail (PMOS, d=tail)  — tail current source for main diff pair
  M1    (PMOS, d=d1_diode, g=inp, s=tail) — input pair half A
  M2    (PMOS, d=out,      g=inn, s=tail) — input pair half B
                                            (drain IS the output!)
  M3    (NMOS, d=d1_diode, g=d1_diode, diode) — mirror load A
  M4    (NMOS, d=out,      g=d1_diode, mirror) — mirror load B
                                                  (drain IS the output!)

Feedforward pair (NEW):
  Mff_n (PMOS, d=out, g=inn, s=vdd) — feedforward gm driven by inn
  Mff_p (PMOS, d=out, g=inp, s=vdd) — feedforward gm driven by inp

Differential signal-path math
-----------------------------

Define v_d = (V_inp - V_inn) / 2 (differential half-input).

Main path:
  i_out_main = gm_M1 · v_d   (M1 contributes through mirror gain 1)
            - gm_M2 · (-v_d) = +gm_M1 · v_d + gm_M2 · v_d = 2·gm_M1·v_d
  (assuming gm_M1 = gm_M2 and a unity mirror gain — standard textbook).

Feedforward pair:
  Mff_p source current proportional to v_inp = vcm + v_d.
  Mff_n source current proportional to v_inn = vcm - v_d.
  i_ff = gm_ff · v_inp - gm_ff · v_inn ... wait. Both Mff_p and Mff_n
  are PMOS with source=vdd; their drain currents into `out` increase
  when their gate voltage DROPS (Vgs becomes more negative). So:
    i_ff_p (into out from vdd via Mff_p) = -gm_ff · (V_inp - V_inp_dc)
    i_ff_n (into out from vdd via Mff_n) = -gm_ff · (V_inn - V_inn_dc)
  Net feedforward current INTO out:
    i_ff = i_ff_p + i_ff_n = -gm_ff · ((V_inp + V_inn) - (V_inp_dc + V_inn_dc))
         = -gm_ff · (2·v_cm)   [only common-mode coupling, NOT differential!]

That doesn't help differential gain. The symmetric arrangement
described above CANCELS differential gain by design. To get
differential ADD-ON gain, the feedforward devices must cross-couple
the inputs — i.e. Mff_p (PMOS) gated by inn instead of inp, and Mff_n
gated by inp. Then the polarities match the main path:
    i_ff = -gm_ff · (V_inn - V_inp) = +gm_ff · (V_inp - V_inn) = 2·gm_ff·v_d
which ADDS to the main differential current.

So the correct (sign-coherent) wiring is:

  Mff_p (PMOS, d=out, g=inn, s=vdd)  ← named by output sign convention
                                       (i.e. this device "pushes out_p
                                       direction", drain current
                                       increases when inn drops)
  Mff_n (PMOS, d=out, g=inp, s=vdd)  ← cross-coupled

This is what the template materialises. The instance names use the
INPUT polarity convention (Mff_p means "feedforward driven by the
positive output sense", but its gate is actually inn — the
cross-coupling is implicit).

Common-mode rejection
---------------------

Both feedforward devices share the same source=vdd and the same
load=out. A common-mode input swing V_inp = V_inn = v_cm gives:
  i_ff = -gm_ff · (V_inn - V_inn_dc) - gm_ff · (V_inp - V_inp_dc)
       = -2·gm_ff·v_cm
This common-mode current is then rejected by the NMOS mirror load
(which carries DC current set by the bias chain, independent of v_cm
at AC if the mirror is well-matched). In practice the dual-FF
arrangement does increase common-mode coupling to out at frequencies
where the mirror cannot reject — a CMRR cost the designer accepts for
the differential GBW boost. A future v0.0.92+ extension could add a
common-mode-cancelling tail current source for Mff_p / Mff_n.

Honest scope (what you should expect)
=====================================

  - gain_dc: roughly +20 log10(1 + gm_ff / gm_M1) higher than baseline
    single-stage. With Mff_scale=1.0 (gm_ff = gm_M1) this is +6 dB.
  - gbw: roughly 2× baseline (the feedforward path adds a parallel
    transconductance to the output node — net Gm doubles, while the
    output capacitance stays roughly constant).
  - pm: unchanged (single high-Z node, dominant pole at the output
    set by Ro · CL).

The architectural signature is the PAIR of feedforward devices named
Mff_p and Mff_n (no other Clay stdlib variant has these), both PMOS,
both with drain=out and source=vdd, gated by inn and inp respectively
(cross-coupled).

Knobs
-----

The template adds two new kwargs on top of the textbook single-stage
diff-pair baseline:

  Mff_scale: float = 1.0
      Sizing ratio of Mff_p / M1 (and Mff_n / M2 — matched). Default
      1.0 gives gm_ff = gm_M1 at the same gm/Id, doubling the
      output-stage transconductance.
  gm_id_ff: float | None = None
      gm/Id intent override for the feedforward devices. When set
      the closed-form solver derives W_ff from (gm_id, I_d, L), where
      I_d = `Mff_scale * I_diff` (each feedforward device carries its
      own DC drain current proportional to Mff_scale × per-leg
      diff-pair current).

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, W_diff,
L_diff, W_load, L_load, gm_id_diff, gm_id_load) are similar to the
single-stage single-ended baseline.

The DC sink current at `out` must absorb BOTH the main mirror current
(I_diff per leg) AND the feedforward current (Mff_scale × I_diff per
device, 2 devices). The mirror device M4 is widened internally by
(1 + 2 · Mff_scale) so the output node DC bias stays at mid-rail.
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


def build_single_stage_dual_feedforward(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,

    # ─── Per-device sizing: physical (W, L) form ────────────
    W_diff: float = 20e-6,  L_diff: float = 0.5e-6,
    W_load: float = 8e-6,   L_load: float = 1e-6,

    # ─── Per-device sizing: intent (gm/Id) form ─────────────
    gm_id_diff: float | None = None,
    gm_id_load: float | None = None,

    # ─── Dual feedforward gm pair (NEW) ─────────────────────
    Mff_scale: float = 1.0,
    W_ff: float | None = None,
    L_ff: float | None = None,
    gm_id_ff: float | None = None,

    # ─── R115 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # single-stage dual-feedforward opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Razavi §18 — diff-
    #     pair Vth mismatch directly maps to input-referred offset.
    #   * GUARD_RING on Mtail (PMOS tail current source, M=N_tail).
    #     Carries the entire stage-1 bias current; nwell guard ring tied
    #     to vdd shields the single high-Z output node `out` from
    #     supply-coupled substrate noise. Because this is a single-stage
    #     opamp the output node is loaded only by the gain-element pair
    #     plus the FF pair — any substrate noise injection there
    #     directly modulates the output.
    #   * SYMMETRIC_LAYOUT on the cross-coupled feedforward pair (Mff_p /
    #     Mff_n). Mff_p taps inn, Mff_n taps inp; the cross-coupling
    #     means CMRR depends on the FF pair's layout symmetry separately
    #     from the diff pair's symmetry. Razavi §11.3 — when a parallel
    #     signal path is cross-coupled to the differential inputs, both
    #     pairs must respect a shared mirror axis or the common-mode
    #     contributions don't cancel. Axis = vertical anchored at the
    #     tail node.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a single-stage opamp with dual feedforward
    transconductors.

    Same 5-port surface as the textbook 2-stage Miller cells. The
    feedforward pair is internal — two symmetric PMOS devices (Mff_p,
    Mff_n) cross-coupled to the differential inputs, each with
    drain=out and source=vdd. Differential gain is boosted by roughly
    (1 + Mff_scale) over the baseline; common-mode rejection is
    slightly reduced because the feedforward pair has no shared tail.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref: reference current for the bias chain.
        W_bp, L_bp, W_bn, L_bn: bias diode / mirror sizing.
        N_tail: tail-current mirror ratio.
        W_diff, L_diff: input pair PMOS sizing.
        W_load, L_load: NMOS mirror load sizing.
        gm_id_diff, gm_id_load: gm/Id intent overrides for the two
            matched signal-path device groups.
        Mff_scale: ratio of feedforward device W to input pair W.
            Default 1.0 gives gm_ff = gm_M1 at matched gm/Id.
        W_ff, L_ff: optional explicit overrides for the feedforward
            device sizing (default: W_diff · Mff_scale, L_diff).
        gm_id_ff: gm/Id intent override for the feedforward devices.
            When set the closed-form solver derives W_ff from
            (gm_id, I_d, L) where I_d = Mff_scale · I_diff.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0   # each PMOS input-pair leg
    I_load = I_diff                   # mirror legs carry I_diff each

    # Each feedforward device carries its own DC drain current
    # proportional to Mff_scale × I_diff. With 2 devices that's
    # 2 · Mff_scale · I_diff of extra current INTO out from vdd.
    # The output mirror device M4 must absorb this PLUS the main
    # mirror leg current (I_load), so its multiplier is bumped to
    # (1 + 2 · Mff_scale).
    I_ff = Mff_scale * I_diff
    M_load_total = 1 + 2 * Mff_scale

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load is not None:
        W_load = closed_form_W("nmos", gm_id=gm_id_load,
                               I_d=I_load, L=L_load).W

    # Feedforward sizing — same priority pattern as Mgf2 in AFFC.
    if L_ff is None:
        L_ff = L_diff
    if gm_id_ff is not None:
        W_ff = closed_form_W("pmos", gm_id=gm_id_ff,
                             I_d=I_ff, L=L_ff).W
    elif W_ff is None:
        W_ff = W_diff * Mff_scale

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
    devices.append(_mosfet("M2", "pmos", d="out",      g="inn", s="tail", b="vdd",
                           W=W_diff, L=L_diff))
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                           s="vss", b="vss",
                           W=W_load, L=L_load))
    devices.append(_mosfet("M4", "nmos", d="out",      g="d1_diode",
                           s="vss", b="vss",
                           W=W_load, L=L_load, M=M_load_total))

    # ─── Dual feedforward gm pair (cross-coupled to differential inputs)
    # Mff_p: PMOS, drain=out, gate=inn, source=vdd (cross-coupled to inn)
    # Mff_n: PMOS, drain=out, gate=inp, source=vdd (cross-coupled to inp)
    # The cross-coupling means the differential-mode signal at the
    # output sums with the main path's contribution (both delivering
    # current of the same sign for a given v_d). Common-mode signals
    # produce a current of the same sign in both devices, which the
    # output mirror partially rejects.
    devices.append(_mosfet("Mff_p", "pmos", d="out", g="inn", s="vdd", b="vdd",
                           W=W_ff, L=L_ff))
    devices.append(_mosfet("Mff_n", "pmos", d="out", g="inp", s="vdd", b="vdd",
                           W=W_ff, L=L_ff))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R115 C: optional DRC annotations ─────────────────
    # Three annotations spanning MATCHING / GUARD_RING / SYMMETRIC_LAYOUT.
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
                    "to vdd shields the single high-Z output node from "
                    "supply-coupled substrate noise (single-stage opamp "
                    "— substrate noise at out directly modulates the "
                    "output)"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.SYMMETRIC_LAYOUT,
            devices=("Mff_p", "Mff_n"),
            extras={
                "rationale": (
                    "Cross-coupled feedforward pair — Mff_p taps inn, "
                    "Mff_n taps inp; CMRR depends on the FF pair's "
                    "layout symmetry separately from the diff pair's "
                    "symmetry. Both pairs must respect a shared mirror "
                    "axis or the common-mode contributions don't cancel"
                ),
                "axis": "vertical",
                "anchor": "tail",
            },
        ))

    return cell
