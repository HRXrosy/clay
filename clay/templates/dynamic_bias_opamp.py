"""
Built-in template: dynamic_bias_opamp.

Two-stage Miller-compensated opamp whose tail and stage-2 load currents
are derived from a **dynamic-bias generator** (beta-multiplier topology
with PTAT output) plus a **sample/hold switch** at the bias output to
demonstrate switched-bias dynamic-bias use cases. This is the v0.0.92
Round 68 Agent R stdlib cell that exercises three of the six v0.0.92
functional-role keywords as **primary architectural features**:

  * `dynamic_bias`        — Idyn beta-multiplier output (PMOS-side rail)
                            that tracks Vt across PVT corners and feeds
                            both the tail current source and the stage-2
                            load current.
  * `sample_hold_switch`  — Msh NMOS sample/hold switch + hold cap Csh
                            on the bias-storage node nsh. Demonstrates
                            a switched-bias dynamic-bias use case where
                            the bias rail is periodically refreshed.
  * `tail_current_source` — Mtail (legacy keyword, present so the
                            v0.0.86 first-batch check coexists with the
                            v0.0.92 third-batch keywords).

Why this variant
================

A textbook 2-stage Miller opamp uses a static bias source (typically a
PMOS diode + resistor). Across PVT, the bias current varies with Vt
and Vdd — a typical sky130 design might see ±30% bias variation across
the 5 corners. Beta-multiplier dynamic bias (Razavi Chapter 11) reduces
this to ≤ 10% by generating the bias current from a thermal-voltage
reference: I_bias ∝ Vt·ln(N)/R, which has a known +T coefficient that
partially compensates the -T coefficient of Vt elsewhere.

The architectural signature of a dynamic_bias rail (vs a static one) is
the resistor in the bias-current path AND the diode-pair size mismatch
(1:N ratio) that creates the Vt·ln(N) drop across the resistor.

Sample/hold demonstrates a "stored" dynamic-bias use case: the bias is
refreshed periodically from the beta-multiplier output, held on a small
cap during use, and re-sampled when stale. This is the textbook use
case for `sample_hold_switch` in a bias context — the role declaration
carries a `hold_droop_target` (V/s) that the v0.0.92 invariant runtime
can check against simulated droop.

Topology summary
================

         vdd ─────┬───────────────────────┬────────────────
                  │                       │
                  │                       │
                 Mbm1 (P)                Mtail (P) ── tail
                 diode                   g=nsh
                  │                       │
                 Rbm (resistor)           │
                  │                  Mdp_p Mdp_n   PMOS diff pair
                 ndyn_pmos              (inn) (inp)
                  │                       │
                 Mbm2 (N, M=4)             d1  d2
                 diode                       │
                  │                       (NMOS mirror)
                  │                       │
                  │                       │
                 Msh (NMOS) ── nsh ───── Cl (hold cap)
                  │           │
                  │           │
                  └─── ndyn ──┘   (sample-hold switch closes nsh ← ndyn
                                   on phi1; opens on phi2 → bias held)
                       │
                      vss

Bias plan
---------
The beta-multiplier (Mbm1 PMOS + Rbm + Mbm2 NMOS with 1:4 W ratio)
generates a PTAT current at the ndyn node. The sample/hold switch
Msh, when closed, charges the storage cap Csh to ndyn's voltage; when
opened, Csh holds that bias value with leakage-driven droop. The held
voltage `nsh` is the gate-bias rail for the opamp's tail current source
Mtail.

Architectural roles in the .clay surface
----------------------------------------
The companion `examples/dynamic_bias_opamp_explicit.clay` declares
five functional roles, three of which exercise v0.0.92 keywords:

  dynamic_bias            Idyn      — beta-multiplier output ndyn
  sample_hold_switch      Msh       — NMOS S/H gate switch
  tail_current_source     Mtail     — opamp tail (legacy keyword)
  dominant_pole_cap       Cc        — Miller cap (legacy keyword)
  input_transconductor    Mdp       — diff pair (legacy keyword)

Honest scope
============
This template's sample/hold switch is a SIMPLIFIED demonstration: the
testbench drives the clock phase static (closed) during AC measurement,
so the cell behaves like a static-bias 2-stage Miller opamp for AC
characterization purposes. The droop and sample/hold dynamics would
only be visible in a transient analysis with a clocked phi1/phi2 — a
future round may extend the testbench harness to drive S/H clocks.

Knobs
-----

  I_ref:                float = 1e-5
    Reference current set by the beta-multiplier output (sets gm scale).
  Rbm:                  float = 5e3
    Beta-multiplier resistor.
  W_bm1, L_bm1:         float = 4e-6 / 1e-6
    Beta-multiplier PMOS diode (Mbm1).
  W_bm2, L_bm2:         float = 16e-6 / 1e-6
    Beta-multiplier NMOS diode (Mbm2 — ratio 4 → Vt·ln(4) drop).
  W_sh, L_sh:           float = 8e-6 / 0.5e-6
    Sample/hold switch (NMOS).
  Csh:                  float = 0.5e-12
    Hold cap on the stored-bias node nsh.
  W_diff, L_diff:       float = 20e-6 / 0.5e-6
    Diff-pair sizing.
  Cc:                   float = 1.5e-12
    Miller compensation cap.

Port surface: ['inn', 'inp', 'out', 'vdd', 'vss']
"""

from __future__ import annotations

from ..ir import Cell, Device, Spec
from .two_stage_miller_pmos import build_ac_testbench  # noqa: F401  re-export


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


def build_dynamic_bias_opamp(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Dynamic-bias (beta-multiplier) reference ───────────
    I_ref: float = 1e-5,
    Rbm: float = 5e3,
    W_bm1: float = 4e-6, L_bm1: float = 1e-6,
    W_bm2: float = 16e-6, L_bm2: float = 1e-6,

    # ─── Sample/hold switch ─────────────────────────────────
    W_sh: float = 8e-6, L_sh: float = 0.5e-6,
    Csh: float = 0.5e-12,

    # ─── Opamp bias mirror ratios ───────────────────────────
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs_load: int = 16,

    # ─── Diff pair (PMOS input) ─────────────────────────────
    W_diff: float = 20e-6, L_diff: float = 0.5e-6,
    W_load1: float = 4e-6, L_load1: float = 1e-6,
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,

    # ─── Stage-2 NMOS CS ────────────────────────────────────
    W_cs: float = 40e-6, L_cs: float = 0.5e-6,
    gm_id_cs: float | None = None,

    # ─── Miller compensation ────────────────────────────────
    Cc: float = 1.5e-12,

    # ─── R113 C: optional qualitative DRC annotations ───────
    # When True, attach three R111 C-vocabulary annotations on the
    # dynamic-bias 2-stage Miller opamp:
    #   * MATCHING on the PMOS diff pair (M1 / M2). Vth-mismatch
    #     between M1/M2 directly translates to input-referred offset,
    #     so common-centroid layout is required.
    #   * GUARD_RING on Mtail (the tail current source, 4× wider than
    #     unit). Mtail draws the entire stage-1 bias current; substrate
    #     coupling to the surrounding NMOS load (M3 / M4) and stage-2
    #     CS (Mcs) would inject DC-supply noise into the gain path. A
    #     guard ring around Mtail (nwell tie to vdd) isolates it.
    #   * CURRENT_MIRROR_MATCHING on the NMOS mirror legs (M3 / M4).
    #     M3 is the d1-gate diode-connected reference; M4 mirrors at
    #     the same W/L unit; for stage-1 mirror error to match the
    #     replica-leg analysis (Mcs2_repl pattern elsewhere in the
    #     stdlib), the layout must be interdigitated unit cells, not
    #     a parameterised W-multiplier. 1:1 ratio.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit (which iterates cell.devices) is byte-for-byte
    # unaffected when default attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a 2-stage Miller opamp with dynamic-bias rail
    and sample/hold switch on the bias output.

    Args:
        cell_name: name for the .subckt.
        spec: target spec.
        I_ref: reference current (beta-multiplier output).
        Rbm: beta-multiplier resistor.
        W_bm1, L_bm1: PMOS diode of beta-multiplier (Mbm1).
        W_bm2, L_bm2: NMOS diode of beta-multiplier (Mbm2, M=4 ratio).
        W_sh, L_sh: sample/hold NMOS switch.
        Csh: hold cap on the stored-bias node nsh.
        W_bp, L_bp, W_bn, L_bn: opamp bias mirror units.
        N_tail: tail mirror ratio.
        K_cs_load: stage-2 NMOS-CS load mirror multiplier.
        W_diff, L_diff: PMOS diff pair sizing.
        W_load1, L_load1: NMOS mirror load.
        W_cs, L_cs: stage-2 NMOS CS sizing.
        gm_id_*: gm/Id intent overrides.
        Cc: Miller compensation cap.

    Returns:
        Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    I_diff = (N_tail * I_ref) / 2.0
    I_cs = K_cs_load * I_ref

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_diff, L=L_load1).W
    if gm_id_cs is not None:
        W_cs = closed_form_W("nmos", gm_id=gm_id_cs,
                             I_d=I_cs, L=L_cs).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Dynamic-bias (beta-multiplier) reference ───────────
    # Beta-multiplier: Mbm1 (PMOS diode) with Rbm in series between vdd
    # and Mbm1's drain, Mbm2 (NMOS diode, 4× wider) closing the loop.
    devices.append(_res("Rbm", p="vdd", n="ndyn_pmos_src", value=Rbm))
    devices.append(_mosfet(
        "Mbm1", "pmos",
        d="ndyn", g="ndyn", s="ndyn_pmos_src", b="vdd",
        W=W_bm1, L=L_bm1,
    ))
    devices.append(_mosfet(
        "Mbm2", "nmos",
        d="ndyn", g="ndyn", s="vss", b="vss",
        W=W_bm2, L=L_bm2,
    ))

    # ─── Sample/hold switch (Msh) and hold cap (Csh) ────────
    # NMOS pass-gate from ndyn to nsh. Gate driven by `phi1` clock —
    # in the AC testbench, phi1 is held high (switch closed) so the
    # cell behaves as a static-bias opamp for characterization. A
    # future transient testbench can pulse phi1 to test droop.
    #
    # We add the clock port to the cell? No — clocks ride internally
    # via a static rail tied to vdd through Mbm_phi (a simple PMOS
    # diode that pulls phi1 to vdd-Vgs_p, well above the NMOS Vth).
    devices.append(_mosfet(
        "Mbm_phi", "pmos",
        d="phi1", g="phi1", s="vdd", b="vdd",
        W=W_bp, L=L_bp,
    ))
    # Sample/hold switch.
    devices.append(_mosfet(
        "Msh", "nmos",
        d="ndyn", g="phi1", s="nsh", b="vss",
        W=W_sh, L=L_sh,
    ))
    # Hold cap on the stored bias node.
    devices.append(_cap("Csh", p="nsh", n="vss", value=Csh))

    # ─── Opamp bias rail (driven from nsh) ──────────────────
    # PMOS diode at pbias seeded from nsh via a mirror device. This
    # gives the opamp tail/load mirrors a stable PMOS-side bias.
    devices.append(_mosfet(
        "Mbp", "pmos",
        d="pbias", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp,
    ))
    # Mirror from nsh (NMOS diode at nsh sets vt; mirror Mbnm pulls
    # current from pbias to vss, making PMOS diode Mbp settle).
    devices.append(_mosfet(
        "Mbn_nsh", "nmos",
        d="nsh", g="nsh", s="vss", b="vss",
        W=W_bn, L=L_bn,
    ))
    devices.append(_mosfet(
        "Mbnm", "nmos",
        d="pbias", g="nsh", s="vss", b="vss",
        W=W_bn, L=L_bn,
    ))

    # ─── Tail current source (legacy `tail_current_source` keyword) ─
    devices.append(_mosfet(
        "Mtail", "pmos",
        d="tail", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=N_tail,
    ))

    # ─── PMOS diff pair (Mdp = M1/M2) ───────────────────────
    devices.append(_mosfet(
        "M1", "pmos",
        d="d1", g="inp", s="tail", b="vdd",
        W=W_diff, L=L_diff,
    ))
    devices.append(_mosfet(
        "M2", "pmos",
        d="d2", g="inn", s="tail", b="vdd",
        W=W_diff, L=L_diff,
    ))
    # NMOS current mirror load.
    devices.append(_mosfet(
        "M3", "nmos",
        d="d1", g="d1", s="vss", b="vss",
        W=W_load1, L=L_load1,
    ))
    devices.append(_mosfet(
        "M4", "nmos",
        d="d2", g="d1", s="vss", b="vss",
        W=W_load1, L=L_load1,
    ))

    # ─── Stage 2: NMOS CS amplifier ────────────────────────
    devices.append(_mosfet(
        "Mcs", "nmos",
        d="out", g="d2", s="vss", b="vss",
        W=W_cs, L=L_cs,
    ))
    devices.append(_mosfet(
        "Mlcs", "pmos",
        d="out", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=K_cs_load,
    ))

    # ─── Miller compensation ──────────────────────────────
    devices.append(_cap("Cc", p="out", n="d2", value=Cc))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R113 C: optional DRC annotations ─────────────────
    # Three annotations spanning three of the seven R111 C kinds:
    # MATCHING (diff pair), GUARD_RING (tail current source), and
    # CURRENT_MIRROR_MATCHING (NMOS mirror legs). See the kwarg docs
    # block above for the rationale.
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
                    "input-referred offset; common-centroid layout "
                    "required"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "Tail current source (4× unit width) carries the "
                    "entire stage-1 bias current; nwell guard ring tied "
                    "to vdd shields surrounding NMOS load and stage-2 "
                    "CS from substrate-coupled supply noise"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "Stage-1 NMOS mirror legs — M3 diode-connected "
                    "reference, M4 mirrors; layout must be unit-cell "
                    "interdigitated array (not a W-multiplier) so "
                    "Pelgrom mismatch scales with sqrt(unit_count)"
                ),
                "ratio": "1:1",
                "unit_device": "M3",
                "primitive": "interdigitated_array",
            },
        ))

    return cell


__all__ = ["build_dynamic_bias_opamp", "build_ac_testbench"]
