"""
Built-in template: nested_miller_3stage_feedforward.

Three-stage opamp with nested Miller compensation and a feedforward path.
This is the canonical AnalogGym NMCF benchmark topology — closes the
"3-stage opamp" gap in Clay's stage-template library (G1 in the
AnalogGym validation doc).

Topology (Razavi Chapter 10 / Leung-Mok NMCF)
=============================================

  vdd ───┬────────┬──────────────┬─────────────────┬──────
         │        │              │                 │
        Mbp    Mtail          Mlcs2 (stage 2 load) Mcs3 (stage 3 P CS)
         │     (PMOS)         PMOS,                PMOS,
   pbias─┤                    gate=bias_d2         gate=d2
         │      tail            │                  │
         │       │              d2 ────────┐       out
         │       │              │          │       │
         │      M1 ── M2        Mcs2       │      Mlcs3
         │       │     │       (NMOS,      │      (NMOS sink,
         │     inp   inn        gate=d1)   │       gate=nbias)
   nbias─┤       │     │        │          │       │
         │       └──d1_diode    │          │      vss
         │       │     │        │          │
         │       │    M3      M4           │
         │       │    (diode) (mirror)     │
         │       │     │       │           │
         │       │    vss     vss          │
         │       │     │       │           │
         │       │   d1=stage-1 output     │
         │       │     │                   │
         │       Mbn   │                   │
         │       │     │                   │
   nbias─┤     vss     │                   │
         │             │                   │
         └─────────────┴──── Mbnm (cross mirror) ───
                                           │
                                           │
   Replica-bias for stage 2 load
   -----------------------------
   Mcs2_repl: NMOS like Mcs2 but gate=d1_diode (DC bias, no signal).
              Sinks the SAME current Mcs2 sinks at the DC OP.
   Mbp_d2   : PMOS diode at bias_d2, gate=drain=bias_d2, source=vdd.
              Sources Mcs2_repl's current; sets bias_d2 = vdd − |Vgs_p|.
   Mlcs2    : PMOS load on d2, gate=bias_d2, source=vdd (current mirror
              of Mbp_d2). Pushes the same current Mcs2 sinks → d2 settles
              at a saturation-friendly voltage.

   Compensation caps (nested Miller + feedforward)
   -----------------------------------------------
   Cm1: between d2 and out   (outer nested Miller, wraps stage 3)
   Cm2: between d1 and d2    (inner nested Miller, wraps stage 2)
   Cf:  between d1 and out   (feedforward, bypasses stage 2 — LHP zero)

Bias plan (2-stage Miller PMOS chain + stage-2 replica)
-------------------------------------------------------
- Iref injects I_ref into nbias from vdd inside the cell.
- Mbn (NMOS diode at nbias) absorbs it.
- Mbnm (NMOS mirror) drains I_ref from pbias to vss.
- Mbp (PMOS diode at pbias) supplies that current from vdd; pbias settles
  ≈ vdd − |Vgs_p|.
- Mtail mirrors Mbp at ratio N_tail → I_tail = N_tail · I_ref.
- Mcs2_repl + Mbp_d2 form a replica leg that sets bias_d2 so Mlcs2 sources
  the correct current for whatever Vgs the diff-pair-mirror chain delivers
  to d1. This is the same replica-bias technique used by the FDA template
  (`clay.templates.fully_differential`); without it, mismatches between
  the closed-form gm/Id solver and sky130's actual sub-Vth operation
  cause d2 to rail to vdd before AC stimulation can do anything useful.
- Mlcs3 mirrors Mbn at ratio K_cs3_load → stage-3 sink current.

Polarity (single-ended view, inp = inverting in this internal convention)
-------------------------------------------------------------------------
- inp ↑ → I_M1 ↓ → d1_diode ↓ → M4 sinks less at d1 → d1 ↑.
- d1 ↑ → Mcs2 (NMOS) conducts more → d2 ↓.
- d2 ↓ → |Vgs_Mcs3| ↑ (PMOS) → Mcs3 conducts more → out ↑.
Net: inp ↑ → out ↑ (three inversions through the mirror + two CS stages
cancel to a positive open-loop sign convention).

Why the feedforward path matters
--------------------------------
Without Cf, a 3-stage opamp has three high-impedance nodes, each
contributing a pole. The two nested Miller caps Cm1 and Cm2 split the
poles via standard pole-splitting analysis, but Cm1 also displaces the
stage-3 pole to the right-half-plane (RHP) as a zero — a classic problem
in the textbook nested-Miller scheme. Cf creates a left-half-plane (LHP)
zero that cancels the RHP zero from Cm1, restoring phase margin.
References: Leung & Mok 2001 NMCF analysis, Razavi 10.6.

Known limitations (v0.0.22)
---------------------------
- 3-stage opamps are notoriously hard to stabilize at low compensation
  capacitance. The defaults here (Cm1=1.5pF, Cm2=0.5pF, Cf=0.5pF) target
  the canonical AnalogGym sizing range but may need user tuning for any
  given spec. If you can't hit a target spec, the typical knobs are:
  (a) increase Cm1 (more pole-splitting at stage-3 output);
  (b) tune Cf/Cm1 ratio to align the LHP zero with the second pole;
  (c) reduce K_cs3_load (lower stage-3 current → lower stage-3 pole).
- No nulling resistor in series with either Miller cap. A real
  production design adds Rz1, Rz2 in series with Cm1, Cm2 to provide
  an extra degree of freedom for zero placement. v0.0.22 omits this
  to keep the template short; users can override by adding a series-R
  manually if their spec demands it.
- Stage 3 has no replica bias; we rely on Mlcs3 (NMOS mirror of nbias)
  to sink K_cs3_load · I_ref and Mcs3 to be sized so its DC current
  matches that sink current at the natural V(d2) set by the stage-2
  replica. If V(out) rails despite the stage-2 replica, the typical
  fix is to reduce K_cs3_load or increase W_cs3.
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


def build_nested_miller_3stage_feedforward(
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
    # Set any of these to override the W kwarg above; the closed-form
    # gm/Id solver derives W from (gm_id, I_d, L). Per-device I_d
    # is derived from the bias plan documented at the top.
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_stage2: float | None = None,
    gm_id_stage3: float | None = None,

    # ─── Compensation caps ──────────────────────────────────
    Cm1: float = 1.5e-12,   # outer nested Miller (d2 ↔ out)
    Cm2: float = 0.5e-12,   # inner nested Miller (d1 ↔ d2)
    Cf: float = 0.5e-12,    # feedforward (d1 ↔ out)

    # ─── R112 C: optional qualitative DRC annotations ───────
    # When True, attach a PASSIVE_MATCHING annotation on the nested
    # Miller cap pair (Cm1 outer / Cm2 inner). The two Miller caps set
    # the pole-splitting ratio of the NMCF stage — Hastings Ch. 8
    # requires that the layout primitive for matched fringing caps is
    # a unit-cell array (not a parameterised metal-stack rectangle) so
    # mismatch is bounded by Pelgrom-like scaling on the unit count.
    # The new R111 C PASSIVE_MATCHING kind routes the layout prompt to
    # a passive-specific matching rule (unit-array of MOM unit cells)
    # rather than the MOS common-centroid array rule used by MATCHING.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit (which iterates cell.devices) is byte-for-byte
    # unaffected when default attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a 3-stage NMCF opamp on sky130.

    Same 5-port surface as the 2-stage Miller cells — the additional
    stage and three compensation caps are internal.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref: reference current into the bias chain.
        W_bp, L_bp: PMOS bias-mirror sizing.
        W_bn, L_bn: NMOS bias-mirror sizing.
        N_tail: PMOS tail-mirror ratio (I_tail = N_tail · I_ref).
        K_cs3_load: NMOS mirror ratio for stage-3 sink and PMOS mirror
            ratio for stage-3 source. The stage-2 load ratio is NOT a
            user knob — it is set by the replica-bias leg.
        W_diff, L_diff: PMOS diff-pair sizing (M1/M2).
        W_load1, L_load1: NMOS mirror-load sizing (M3/M4).
        W_cs2, L_cs2: stage-2 NMOS CS sizing (Mcs2). The replica device
            Mcs2_repl is sized identically.
        W_cs3, L_cs3: stage-3 PMOS CS sizing (Mcs3).
        gm_id_diff: intent override for the PMOS diff pair (M1/M2).
        gm_id_load1: intent override for the NMOS mirror load (M3/M4).
        gm_id_stage2: intent override for the stage-2 NMOS CS (Mcs2 +
            replica). The closed-form I_d uses the diff-pair half current
            scaled by the W ratio so the replica bias is consistent.
        gm_id_stage3: intent override for the stage-3 PMOS CS (Mcs3).
        Cm1, Cm2, Cf: nested-Miller and feedforward cap values.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0     # each PMOS diff-pair leg
    I_load1 = I_diff                    # each NMOS mirror leg
    # Stage 2 current: the NMOS gain device Mcs2 has gate = d1 (whose DC
    # voltage equals Vgs_M3 because the load mirror M3/M4 sets it). So
    # Mcs2's actual DC current equals (W_cs2/L_cs2) / (W_load1/L_load1)
    # times I_M3 = I_diff. We use this as the closed-form solver's I_d
    # target. (For the v0.0.22 closed-form square-law solver this is
    # only approximate, since the load mirror often operates in sub-Vth
    # on sky130; the replica-bias leg below cleans up the OP mismatch.)
    I_cs2 = I_diff                     # ≈ I_M3 (refined below for solver)
    I_cs3 = K_cs3_load * I_ref         # stage-3 CS sink current

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
    # Same shape as the 2-stage Miller PMOS bias chain.
    devices.append(_mosfet("Mbp",  "pmos", d="pbias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn",  "nmos", d="nbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbnm", "nmos", d="pbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # Iref injected into nbias from vdd.
    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    # ─── Stage 1: PMOS diff pair with NMOS mirror load ───────
    # Tail current source: PMOS mirror of Mbp, ratio N_tail.
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))

    # PMOS diff pair:
    # - M1 (gate=inp) drains to d1_diode (mirror-diode reference node)
    # - M2 (gate=inn) drains to d1 (high-Z stage-1 output)
    # This convention matches the 2-stage Miller PMOS: the input that drives
    # the diode-side leg is "inp"; the high-Z output is on the "inn" leg.
    devices.append(_mosfet("M1", "pmos", d="d1_diode", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="d1",       g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # NMOS mirror load: M3 diode at d1_diode, M4 mirror at d1.
    devices.append(_mosfet("M3", "nmos", d="d1_diode", g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos", d="d1",       g="d1_diode",
                            s="vss", b="vss",
                            W=W_load1, L=L_load1))

    # ─── Stage 2: NMOS CS amplifier driven by d1 ─────────────
    # Mcs2: NMOS, gate=d1, drain=d2, source=vss. Inverting.
    devices.append(_mosfet("Mcs2", "nmos", d="d2", g="d1", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))

    # ─── Stage 2 replica-bias leg ────────────────────────────
    # Mcs2_repl is a Mcs2 replica with gate tied to d1_diode (= V_GS of M3),
    # which matches the DC voltage of d1 (because M3/M4 form a current
    # mirror, V(d1) = V(d1_diode) in DC). Mcs2_repl therefore sinks the
    # SAME DC current Mcs2 sinks regardless of sub-Vth / strong-inversion
    # regime. Mbp_d2 is a PMOS diode at bias_d2 supplying that current
    # from vdd, setting bias_d2 = vdd − |Vsg_p(I_Mcs2_DC)|.
    devices.append(_mosfet("Mcs2_repl", "nmos",
                            d="bias_d2", g="d1_diode", s="vss", b="vss",
                            W=W_cs2, L=L_cs2))
    devices.append(_mosfet("Mbp_d2", "pmos",
                            d="bias_d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # Mlcs2: PMOS current-source load for stage 2, gate = bias_d2 (mirror
    # of Mbp_d2 with the same W/L → 1:1 ratio). Now Mlcs2 sources the
    # SAME current Mcs2 sinks → d2 settles in saturation regardless of
    # how the closed-form solver biased the diff-pair-mirror chain.
    devices.append(_mosfet("Mlcs2", "pmos",
                            d="d2", g="bias_d2", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))

    # ─── Stage 3: PMOS CS amplifier (output stage) ───────────
    # Mcs3: PMOS, gate=d2, drain=out, source=vdd. Inverting.
    devices.append(_mosfet("Mcs3", "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs3, L=L_cs3))

    # Stage-3 load: NMOS current-sink mirror of Mbn, ratio K_cs3_load.
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load))

    # ─── Compensation: nested Miller caps + feedforward ──────
    # Cm1 is the OUTER nested Miller cap: wraps stage 3 (d2 ↔ out).
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    # Cm2 is the INNER nested Miller cap: wraps stage 2 (d1 ↔ d2).
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))
    # Cf is the FEEDFORWARD cap: d1 → out, bypassing stage 2.
    # Creates an LHP zero that cancels the RHP zero introduced by Cm1.
    devices.append(_cap("Cf",  p="out", n="d1", value=Cf))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R112 C: optional PASSIVE_MATCHING annotation ───────
    # The Cm1 outer / Cm2 inner Miller caps set the NMCF pole-splitting
    # ratio. Both are MOM (or MOMCAP) fringing-cap unit arrays in the
    # layout — Hastings Ch. 8 says the matched primitive is a unit-cap
    # array, not a parameterised metal stack. R111 C PASSIVE_MATCHING
    # kind routes the layout prompt to passive matching rules (unit
    # MOM cell, common-centroid array, edge-cell dummies) rather than
    # the MOS array rules used by MATCHING. Cf is intentionally NOT
    # included: it sits between d1 and out (a different node pair from
    # Cm1's out↔d2 and Cm2's d2↔d1), so it has no matching constraint
    # against Cm1/Cm2 — only ratio against itself if duplicated, which
    # the NMCF baseline does not require.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("Cm1", "Cm2"),
            extras={
                "rationale": (
                    "Nested Miller cap pair — Cm1 / Cm2 set the NMCF "
                    "pole-splitting ratio; matching must be on a unit MOM "
                    "array (Hastings Ch. 8), not on a parameterised metal "
                    "stack, so the Pelgrom mismatch scales with sqrt(unit_count)"
                ),
                "primitive": "fringing_cap_unit_array",
            },
        ))

    return cell
