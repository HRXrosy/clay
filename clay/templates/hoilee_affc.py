"""
Built-in template: nested_miller_3stage_affc.

Three-stage opamp with nested Miller compensation AND an *active*
feedforward path (Active FeedForward Compensation, AFFC). This is the
HoiLee 2003 architecture — the canonical AFFC variant of the NMCF family,
distinguished from the textbook Cf-only NMCF by an additional active
transconductor (gm_f2) that injects current directly into the output node
from the stage-1 output, bypassing stage 2.

Topology delta vs `nested_miller_3stage`
========================================

The signal path inherits the NMCF baseline (PMOS diff pair → NMOS-mirror
load → NMOS CS stage-2 → PMOS CS stage-3 with NMOS sink) and ALL of its
compensation devices (Cm1 outer Miller, Cm2 inner Miller, Cf passive
feedforward cap). It adds ONE new device:

  Mgf2 : PMOS, g=d1, d=out, s=vdd  (W = Mgf2_scale * W_cs3, L = L_cs3)

That is, gm_f2 is a PMOS common-source amplifier sitting in parallel
with the stage-3 output device Mcs3, but driven by *stage 1's output*
(d1) instead of stage 2's output (d2). The DC current it sources into
the output node from VDD is balanced by widening the stage-3 sink
Mlcs3 by the same factor (`K_cs3_load` is multiplied by
`(1 + Mgf2_scale)`).

Schematic (output stage delta only — rest unchanged from NMCF baseline)
-----------------------------------------------------------------------

  vdd ────────┬─────────────┬──────────────┐
              │             │              │
            Mcs3 (g=d2)   Mgf2 (g=d1)      │   <-- NEW: active feedforward
              │             │              │
              └──── out ────┘              │
                    │                      │
                  Mlcs3 (g=nbias, M scaled up to absorb both)
                    │
                   vss

Reference (AnalogGym `Amplifier/spice_netlist/HoiLee_AFFC_Pin_3`):
  xm12 VOUT net050 VDDA VDDA pfet (gmf2_PMOS)
  xm25 VOUT net049 GNDA GNDA nfet (gm3_NMOS)
The Clay analogue is `Mgf2 = xm12` with gate=d1 (≈net050, stage-1
output) and drain=out (=VOUT), source=vdd (=VDDA).

Why the active feedforward path matters (transfer-function level)
-----------------------------------------------------------------

The plain Cf-only NMCF baseline gets one *passive* LHP zero from the
Cf cap (s = -gm3 / Cf, frequency ~ gm3 / (2π Cf)). The AFFC path
adds a second feedforward term proportional to gm_f2 / gm3 — its
location in the s-plane is gm_f2 · g_o3 / (s · C_out) at low
frequencies, but at high frequencies the two parallel signal paths
through the output stage interact: the stage-2 path
(stage1 → stage2 → stage3) accumulates phase delay from both stage-2
and stage-3 poles, while the active feedforward path (stage1 → gm_f2 →
out) only accumulates one pole (Mgf2's own). The interaction creates
an extra LHP zero in the loop transfer function that pushes phase
margin up at higher GBW than is achievable with Cf alone.

In practice for v0.0.50 we expect:
  - DC gain: roughly preserved or slightly reduced (gm_f2 in parallel
    with Mcs3 from a *different* gate increases output-stage
    transconductance but doesn't double DC gain because gm_f2 is
    driven by d1 not d2 — its DC component is gm_f2 · A_stage1
    instead of gm_f2 · A_stage1 · A_stage2). Net effect: the AFFC
    output stage's DC gain ≈ Mcs3's DC gain + Mgf2's DC gain via
    the bypass path, which is typically a few dB different from
    pure NMCF in either direction depending on stage-2 gain.
  - GBW: the feedforward path lowers the effective phase shift at
    crossover, so for the same Cm1 the dominant pole sits at the
    same place but the loop can tolerate higher GBW before phase
    margin collapses. We don't claim GBW *improves* without
    re-tuning Cm1 — the honest claim is that GBW changes (typically
    by 10–30 %) due to the new device altering the load impedance
    at the output node.
  - Distinguishing test: the materialised SPICE netlist contains a
    new device named `Mgf2` that the NMCF baseline never emits.
    This is what `tests/test_hoilee_affc_distinct.py` asserts.

Bias plan
---------

Identical to the NMCF baseline (PMOS pbias / NMOS nbias / replica
bias_d2 leg for stage 2 / NMOS sink mirror for stage 3) with one
modification: the NMOS sink Mlcs3 multiplier is increased so the DC
operating point of the output node stays in the same place as the
baseline. The default `K_cs3_load` is therefore multiplied by
`(1 + Mgf2_scale)` internally before sizing Mlcs3.

The reasoning: at DC, Mgf2 sources current from VDD into the output
node with a magnitude proportional to V(d1). For the small-signal AC
characterisation we care about, V(d1) sits at its DC operating point
(set by the stage-1 mirror), and the DC current through Mgf2 is
approximately (W_Mgf2/W_Mcs3) · I_Mcs3 = Mgf2_scale · K_cs3_load · I_ref.
The NMOS sink must absorb both Mcs3's and Mgf2's DC currents, so its
mirror ratio is bumped by (1 + Mgf2_scale).

Knobs
-----

The template adds two new kwargs on top of the NMCF baseline:

  Mgf2_scale: float = 1.0
      Sizing ratio of Mgf2 / Mcs3. Default 1.0 gives gm_f2 ≈ gm_3
      at the same gm/Id (1× the feedforward transconductance vs the
      main output device). Values in 0.5–2.0 are physical; below 0.5
      the feedforward path is too weak to matter, above 2.0 the
      output stage burns too much extra DC current. The Hoi Lee 2003
      paper uses gm_f2 / gm_3 ≈ 1.0 with separate sizing knobs; the
      AnalogGym sizing variables expose them independently
      (MOSFET_12_1 = gmf2, MOSFET_25_1 = gm3 in the reference netlist).

  W_gf2, L_gf2: optional explicit overrides
      When set, these override the (W_cs3 * Mgf2_scale, L_cs3) defaults
      and let the user size Mgf2 independently of Mcs3. Useful if the
      designer wants gm_f2 with a different L for headroom or
      mismatch reasons.

All other knobs (I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, W_diff, L_diff,
W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3, gm_id_diff, gm_id_load1,
gm_id_stage2, gm_id_stage3, Cm1, Cm2, Cf) are inherited verbatim from
the NMCF baseline — see `clay/templates/nested_miller_3stage.py`.
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


def build_nested_miller_3stage_affc(
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

    # ─── Compensation caps ──────────────────────────────────
    Cm1: float = 1.5e-12,   # outer nested Miller (d2 ↔ out)
    Cm2: float = 0.5e-12,   # inner nested Miller (d1 ↔ d2)
    Cf: float = 0.5e-12,    # passive feedforward cap (d1 ↔ out)

    # ─── AFFC-specific: active feedforward (gm_f2) ──────────
    # Sizing ratio of Mgf2 / Mcs3. Default 1.0 → gm_f2 ≈ gm_3 at
    # the same gm/Id. The stage-3 NMOS sink mirror Mlcs3 is widened
    # internally by (1 + Mgf2_scale) so the DC operating point of
    # the output node is preserved vs the plain NMCF baseline.
    Mgf2_scale: float = 1.0,
    W_gf2: float | None = None,
    L_gf2: float | None = None,

    # gm/Id intent knob for Mgf2 (overrides W_gf2 / L_gf2 like the
    # baseline gm_id_* knobs).
    gm_id_gf2: float | None = None,
    # R109 C: qualitative DRC annotation surface (additive metadata,
    # see folded_cascode.py R108 C precedent). Default False keeps
    # every pre-R109 caller byte-for-byte identical. When True, three
    # qualitative hints attach:
    #   MATCHING       on PMOS diff pair (M1 / M2),
    #   GUARD_RING     on PMOS tail current source (Mtail),
    #   WELL_ISOLATED  on the parallel output PMOS pair (Mcs3 / Mgf2)
    #                  — the AFFC-defining active feedforward shares the
    #                  output drain with Mcs3, and well coupling between
    #                  them feeds output swing back into both gates via
    #                  body-effect.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a HoiLee AFFC 3-stage opamp.

    Same 5-port surface as the 2-stage Miller cells and the NMCF
    baseline — the additional active feedforward device is internal.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref, W_bp, L_bp, W_bn, L_bn, N_tail, K_cs3_load: bias chain
            sizing (see NMCF baseline docstring).
        W_diff, L_diff, W_load1, L_load1, W_cs2, L_cs2, W_cs3, L_cs3:
            per-device physical sizing for the NMCF baseline path.
        gm_id_diff, gm_id_load1, gm_id_stage2, gm_id_stage3:
            gm/Id intent overrides for the four NMCF baseline devices.
        Cm1, Cm2, Cf: nested-Miller and passive-feedforward cap values.
        Mgf2_scale: ratio of Mgf2 W to Mcs3 W (default 1.0).
        W_gf2, L_gf2: optional explicit overrides for the feedforward
            device sizing (default: scaled from W_cs3 / L_cs3).
        gm_id_gf2: gm/Id intent override for Mgf2. When set the closed-form
            solver derives W_gf2 from (gm_id, I_d, L) where I_d is
            `Mgf2_scale * K_cs3_load * I_ref`.

    Returns:
        A Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan (same as NMCF).
    I_diff = (N_tail * I_ref) / 2.0     # each PMOS diff-pair leg
    I_load1 = I_diff                    # each NMOS mirror leg
    I_cs2 = I_diff                      # ≈ I_M3 stage-2 NMOS CS current
    I_cs3 = K_cs3_load * I_ref          # stage-3 PMOS CS current (Mcs3 alone)

    # AFFC: Mgf2 draws additional DC current proportional to Mgf2_scale.
    I_gf2 = Mgf2_scale * K_cs3_load * I_ref

    # NMOS sink at the output must absorb BOTH Mcs3 (= I_cs3) and Mgf2
    # (= I_gf2) currents. Bump the mirror ratio internally so the
    # output operating point stays at the same V(out) as the plain
    # NMCF baseline.
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

    # Mgf2 sizing: gm_id override beats explicit W_gf2 / L_gf2 which
    # beats the Mgf2_scale-derived default.
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

    # ─── Stage 2 replica-bias leg ────────────────────────────
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

    # ─── AFFC: active feedforward device gm_f2 ───────────────
    # PMOS in parallel with Mcs3 but gated by d1 (stage-1 output)
    # instead of d2 (stage-2 output). Bypasses stage 2 — creates the
    # active LHP zero that distinguishes AFFC from textbook NMCF.
    devices.append(_mosfet("Mgf2", "pmos", d="out", g="d1", s="vdd", b="vdd",
                            W=W_gf2, L=L_gf2))

    # Stage-3 NMOS sink (mirror of nbias) — widened to absorb both
    # Mcs3 and Mgf2 DC currents. Internally K_cs3_load_total =
    # K_cs3_load · (1 + Mgf2_scale).
    devices.append(_mosfet("Mlcs3", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs3_load_total))

    # ─── Compensation: nested Miller caps + passive Cf ───────
    # Note: the active gm_f2 path is ADDITIVE on top of the passive Cf
    # cap. The HoiLee 2003 paper keeps both — Cf handles HF cancellation
    # via direct capacitive coupling; gm_f2 handles MF cancellation
    # via the active feedforward zero. Keeping both in v0.0.50 means
    # the AFFC cell strictly extends NMCF rather than replacing one
    # mechanism with another.
    devices.append(_cap("Cm1", p="out", n="d2", value=Cm1))
    devices.append(_cap("Cm2", p="d2",  n="d1", value=Cm2))
    devices.append(_cap("Cf",  p="out", n="d1", value=Cf))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R109 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extends the
    # template inventory consumed by the layout prompt's round-trip path.
    # Annotations live in cell.__dict__["_drc_annotations"] only — SPICE
    # emit (which iterates cell.devices) is unaffected.
    #
    # NOTE on substitution: R109 C plan called for a "cfcc_amplifier.py"
    # template, but the actual CFCC implementation lives in sau_cfcc.py
    # (already covered in this round). hoilee_affc is substituted as the
    # third template — same NMCF-family sibling, distinct architectural
    # marker (active feedforward instead of cascode-indirect Miller).
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). Inherited from
        #    the NMCF baseline; input offset is set by M1 / M2 Vth-mismatch
        #    (Hastings Ch. 8 common-centroid prescription).
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). Same
        #    rationale as folded_cascode / sau_cfcc / peng_tcfc: high-Z
        #    tail node carries N_tail × I_ref and substrate coupling
        #    converts to common-mode disturbance.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — AFFC output-stage parallel PMOS pair
        #    (Mcs3 / Mgf2). This is the architectural marker of the
        #    Hoi Lee AFFC topology: two PMOS devices both driving `out`
        #    from vdd, one (Mcs3) gated by d2 and one (Mgf2) gated by d1.
        #    If they share an nwell, output swing couples through the
        #    body-effect onto both bulks simultaneously, modulating
        #    |Vth_p| of BOTH devices in lockstep and reducing the
        #    distinct-gm advantage of the active feedforward path. A
        #    dedicated nwell tie (each in its own well, body=vdd)
        #    decouples the two gm transconductors as the AFFC analysis
        #    assumes.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcs3", "Mgf2"),
            extras={
                "rationale": "AFFC parallel PMOS pair — body-effect decoupling between active feedforward gm and stage-3 gm",
                "isolate_from": ("Mload1_p", "Mlcs2"),
            },
        ))

    return cell
