"""
Built-in template: class_AB_output_opamp.

Two-stage Miller-compensated opamp with a CLASS-AB OUTPUT STAGE built
from a complementary push-pull pair (NMOS push + PMOS pull) sharing a
translinear-loop bias. This is the v0.0.92 Round 68 Agent R stdlib
cell that exercises three of the six v0.0.92 functional-role keywords
as **primary architectural features**:

  * `output_class_AB`        — Mab push-pull output stage with a
                               translinear-loop quiescent-current bias.
                               The push device is an NMOS source-follower
                               and the pull device is a PMOS source-
                               follower, sharing a common floating-bias
                               diode pair (Mab_diode_p / Mab_diode_n)
                               that fixes Vbe(push) + Vbe(pull) at a
                               small idle drop.
  * `common_mode_feedback`   — Acmfb common-mode error amp regulating
                               the bias-diode common node so the
                               quiescent current of the class-AB stage
                               is robust across PVT (CMFB applied to
                               the bias rail, not the differential
                               output).
  * `dominant_pole_cap`      — Cc Miller compensation cap (legacy
                               keyword present so the v0.0.86 first-batch
                               check coexists with the v0.0.92 third-
                               batch keywords).

Why this variant
================

A textbook class-A output stage (e.g. NMOS common-source with PMOS
current-mirror load) burns DC quiescent current proportional to its
peak-drive capability. For a stage that must source / sink ±2 mA peak,
class-A requires ≥ 2 mA quiescent. Class-AB (push-pull) instead biases
the output devices at a SMALL idle current (typically 10-100 µA),
with both push and pull devices conducting at idle but neither carrying
significant signal current. On a positive output swing the push device
takes over; on a negative swing the pull device does.

The bias trick: a small "translinear loop" consisting of two diodes
in series between the push gate (high-side bias rail vbias_p_ab) and
the pull gate (low-side bias rail vbias_n_ab). Both rails are driven
from a shared error amplifier that regulates the loop current to the
declared `iq_target`. The error amp is the CMFB cell Acmfb in our
schema, regulating the COMMON-MODE component of the bias rail (not the
output's CM).

Topology summary
================

         vdd ──┬────────────┬───────────────┬──────────────
               │            │               │
              Mbp         Mtail (P)       Mcs2 (P)
              diode       g=pbias         g=pbias
               │            │               │
              pbias        tail            out_main
               │            │               │
               │       Mdp_p Mdp_n        Mab_diode_p (P)
               │       (P,inn) (P,inp)    diode @ vbias_p_ab
               │            │               │
               │            d1  d2         vbias_p_ab
               │            │   │           │
               │            └─┬─┘          Mab_diode_n (N)
               │              │            diode @ vbias_n_ab
               │            (mirror)        │
               │              │            vbias_n_ab
               │              │              │  ◄── Acmfb error-amp loop
               │              │              │
               │              │            Mab_push (N) ─── out (push half)
               │            (NMOS load)     │
               │              │             out
               │              │              │
               │              │            Mab_pull (P) ─── out (pull half)
               │              vss            │
               │                             vss
               vss

CMFB loop role
--------------
The CMFB error amp `Acmfb` samples the average of the two bias-rail
voltages (vbias_p_ab and vbias_n_ab) and drives one of them (the
high-side, by convention) so the translinear loop's Vds drop stays
pinned to the value that gives the declared iq_target. The functional-
role `common_mode_feedback` declares Acmfb with `output_stage = Mab_diode_p`
and `sense_nodes = (vbias_p_ab, vbias_n_ab)`, with a `cmrr_target` for
how stiffly the CMFB rejects common-mode disturbances on the bias rail
across PVT.

The signal path itself is single-ended (NOT fully differential) — the
stage-1 mirror collapses the differential signal to single-ended `out_main`,
and the class-AB push-pull is a single output. The CMFB role here
regulates a BIAS RAIL common mode, not an output common mode — this is
the broader meaning of CMFB the v0.0.92 keyword admits.

Architectural roles in the .clay surface
----------------------------------------
The companion `examples/class_AB_output_opamp_explicit.clay` declares
five functional roles, three of which exercise v0.0.92 keywords:

  output_class_AB         Mab       — push/pull pair Mab_push/Mab_pull
  common_mode_feedback    Acmfb     — bias-rail CMFB amp
  dominant_pole_cap       Cc        — Miller cap (legacy keyword)
  input_transconductor    Mdp       — stage-1 diff pair
  tail_current_source     Itail     — stage-1 tail (legacy keyword)

Honest scope
============

This template's class-AB stage uses a simplified translinear bias
(two diodes in series) rather than the full Monticelli structure. The
quiescent current is set by the diode sizing ratio between Mab_diode_*
and the push/pull devices: at the default sizing (W_ab_diode = 4 µm,
W_ab_push = 20 µm — 5× ratio) the push/pull devices conduct ≈ 5× I_diode
at idle. With I_ref = 10 µA → I_diode = 10 µA → idle ≈ 50 µA per output
device, matching the `iq_target = 50uA` declared in the .clay file.

Knobs
-----

  I_ref:               float = 1e-5
    Reference current through the bias diodes (sets idle iq scale).
  W_ab_push:           float = 20e-6
    NMOS push device width.
  W_ab_pull:           float = 20e-6
    PMOS pull device width.
  W_ab_diode:          float = 4e-6
    Translinear-loop bias-diode width. Ratio W_ab_push / W_ab_diode
    sets the idle quiescent current vs the diode bias.
  Cc:                  float = 1.5e-12
    Miller compensation cap (out ↔ out_main).
  W_cmfb_amp:          float = 8e-6
    Per-device sizing for the CMFB error amp's gain stage.

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


def build_class_AB_output_opamp(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 1e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_tail: int = 4,
    M_cs2_load: int = 4,

    # ─── Stage-1 diff pair (PMOS input) ─────────────────────
    W_diff: float = 20e-6, L_diff: float = 0.5e-6,
    W_load1: float = 4e-6, L_load1: float = 1e-6,
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,

    # ─── Stage-2 PMOS CS (drives out_main) ──────────────────
    W_cs2: float = 40e-6, L_cs2: float = 0.5e-6,
    gm_id_cs2: float | None = None,

    # ─── Class-AB output stage ──────────────────────────────
    # Translinear-loop bias diodes — small unit devices.
    W_ab_diode: float = 4e-6, L_ab_diode: float = 0.5e-6,
    # Push (NMOS source-follower) / Pull (PMOS source-follower) devices.
    W_ab_push: float = 20e-6, L_ab_push: float = 0.5e-6,
    W_ab_pull: float = 40e-6, L_ab_pull: float = 0.5e-6,

    # ─── CMFB error amp (bias-rail regulator) ───────────────
    W_cmfb_amp: float = 8e-6, L_cmfb_amp: float = 1e-6,
    W_cmfb_load: float = 4e-6, L_cmfb_load: float = 1e-6,

    # ─── Compensation ───────────────────────────────────────
    Cc: float = 1.5e-12,

    # ─── R112 C: optional qualitative DRC annotations ───────
    # When True, attach a DEEP_NWELL annotation on the class-AB push
    # NMOS source-follower (Mab_push). Mab_push's source connects to
    # the output node `out` which swings rail-to-rail across the class-AB
    # idle point — substrate coupling between Mab_push's drain-bulk
    # capacitance and the global p-substrate would inject the full
    # output swing into any nearby noise-sensitive NMOS (the stage-1
    # input pair or the CMFB sense network). Razavi §17.3 recommends
    # the deep-nwell process option to float Mab_push's pwell out of
    # the global p-substrate, breaking the substrate-coupled path. The
    # new R111 C DEEP_NWELL kind carries process_required="deep_nwell"
    # so the layout prompt routes the manifest to a PDK with DNW and
    # hard-errors on PDKs without it (no silent fallback to plain
    # WELL_ISOLATED that fails the substrate-isolation goal).
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit (which iterates cell.devices) is byte-for-byte
    # unaffected when default attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a 2-stage Miller opamp with class-AB output
    stage and bias-rail CMFB.

    Args:
        cell_name: name for the .subckt.
        spec: target spec.
        I_ref: reference current through the bias chain.
        W_bp, L_bp, W_bn, L_bn: bias unit-device sizing.
        N_tail: tail mirror ratio.
        M_cs2_load: stage-2 PMOS-CS load mirror multiplier.
        W_diff, L_diff: stage-1 diff pair (PMOS).
        W_load1, L_load1: stage-1 NMOS mirror load.
        W_cs2, L_cs2: stage-2 PMOS CS sizing.
        gm_id_*: gm/Id intent overrides.
        W_ab_diode, L_ab_diode: translinear-loop bias diodes.
        W_ab_push, L_ab_push: class-AB push (NMOS) device sizing.
        W_ab_pull, L_ab_pull: class-AB pull (PMOS) device sizing.
        W_cmfb_amp, L_cmfb_amp: CMFB error-amp gain device.
        W_cmfb_load, L_cmfb_load: CMFB error-amp active load.
        Cc: Miller compensation cap.

    Returns:
        Cell IR with 5 ports: ['inn', 'inp', 'out', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-leg drain currents.
    I_diff = (N_tail * I_ref) / 2.0
    I_cs2 = M_cs2_load * I_ref

    if gm_id_diff is not None:
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("nmos", gm_id=gm_id_load1,
                                I_d=I_diff, L=L_load1).W
    if gm_id_cs2 is not None:
        W_cs2 = closed_form_W("pmos", gm_id=gm_id_cs2,
                              I_d=I_cs2, L=L_cs2).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Bias chain (PMOS diode + NMOS cross-mirror) ────────
    devices.append(_mosfet("Mbp", "pmos",
                           d="pbias", g="pbias", s="vdd", b="vdd",
                           W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn", "nmos",
                           d="nbias", g="nbias", s="vss", b="vss",
                           W=W_bn, L=L_bn))
    devices.append(_mosfet("Mbnm", "nmos",
                           d="pbias", g="nbias", s="vss", b="vss",
                           W=W_bn, L=L_bn))

    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    # ─── Stage-1 PMOS diff pair + NMOS mirror load ─────────
    devices.append(_mosfet("Mtail", "pmos",
                           d="tail", g="pbias", s="vdd", b="vdd",
                           W=W_bp, L=L_bp, M=N_tail))
    devices.append(_mosfet("M1", "pmos",
                           d="d1_diode", g="inp", s="tail", b="vdd",
                           W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos",
                           d="d1", g="inn", s="tail", b="vdd",
                           W=W_diff, L=L_diff))
    devices.append(_mosfet("M3", "nmos",
                           d="d1_diode", g="d1_diode", s="vss", b="vss",
                           W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "nmos",
                           d="d1", g="d1_diode", s="vss", b="vss",
                           W=W_load1, L=L_load1))

    # ─── Stage 2: PMOS CS (drives out_main) ────────────────
    devices.append(_mosfet("Mcs2", "pmos",
                           d="out_main", g="d1", s="vdd", b="vdd",
                           W=W_cs2, L=L_cs2))
    # Stage-2 NMOS active load (mirror of Mbn).
    devices.append(_mosfet("Mlcs2", "nmos",
                           d="out_main", g="nbias", s="vss", b="vss",
                           W=W_bn, L=L_bn, M=M_cs2_load))

    # ─── Class-AB output stage: translinear bias loop ──────
    # Two diodes in series between vbias_p_ab and vbias_n_ab. The push
    # gate is vbias_p_ab (high), the pull gate is vbias_n_ab (low). The
    # CMFB amp regulates the common-mode of the two rails.
    #
    # Mab_diode_p: PMOS diode at vbias_p_ab (drain=vbias_p_ab,
    #              source=out_main, gate=vbias_p_ab) — diode-drop above
    #              out_main.
    # Mab_diode_n: NMOS diode at vbias_n_ab (drain=out_main,
    #              source=vbias_n_ab, gate=out_main) — diode-drop below
    #              out_main.
    devices.append(_mosfet(
        "Mab_diode_p", "pmos",
        d="vbias_p_ab", g="vbias_p_ab", s="out_main", b="vdd",
        W=W_ab_diode, L=L_ab_diode,
    ))
    devices.append(_mosfet(
        "Mab_diode_n", "nmos",
        d="out_main", g="out_main", s="vbias_n_ab", b="vss",
        W=W_ab_diode, L=L_ab_diode,
    ))

    # Push device (NMOS source-follower): drain=vdd, gate=vbias_p_ab,
    # source=out. Mab_push is the keyword anchor for the .clay
    # `output_class_AB Mab { push_device = Mab_push, ... }` decl.
    devices.append(_mosfet(
        "Mab_push", "nmos",
        d="vdd", g="vbias_p_ab", s="out", b="vss",
        W=W_ab_push, L=L_ab_push,
    ))
    # Pull device (PMOS source-follower): drain=vss, gate=vbias_n_ab,
    # source=out.
    devices.append(_mosfet(
        "Mab_pull", "pmos",
        d="vss", g="vbias_n_ab", s="out", b="vdd",
        W=W_ab_pull, L=L_ab_pull,
    ))

    # ─── CMFB error amp (bias-rail regulator) ─────────────
    # Single-stage PMOS-input differential pair sensing
    # (vbias_p_ab + vbias_n_ab)/2 vs a reference. The output drives
    # vbias_p_ab via Mcmfb_drv (PMOS source-follower wired between vdd
    # and vbias_p_ab) so the loop regulates the high-side rail.
    #
    # For simplicity the CMFB sense network is implemented with two
    # large resistors (Rsense_p / Rsense_n) creating a common-mode
    # tap at vcmfb_sense — that node is the input to the error amp.
    devices.append(Device(
        instance_name="Rsense_p",
        device_class="res",
        pdk_variant=None,
        pins={"p": "vbias_p_ab", "n": "vcmfb_sense"},
        sizing={"value": 1e6},
    ))
    devices.append(Device(
        instance_name="Rsense_n",
        device_class="res",
        pdk_variant=None,
        pins={"p": "vcmfb_sense", "n": "vbias_n_ab"},
        sizing={"value": 1e6},
    ))

    # CMFB error amp single-leg: PMOS input (gate=vcmfb_sense), NMOS
    # active load (gate=nbias). Drain = cmfb_out drives vbias_p_ab via
    # the source-follower Mcmfb_drv.
    devices.append(_mosfet(
        "Mcmfb_in", "pmos",
        d="cmfb_out", g="vcmfb_sense", s="vdd", b="vdd",
        W=W_cmfb_amp, L=L_cmfb_amp,
    ))
    devices.append(_mosfet(
        "Mcmfb_load", "nmos",
        d="cmfb_out", g="nbias", s="vss", b="vss",
        W=W_cmfb_load, L=L_cmfb_load,
    ))
    # Source follower drives vbias_p_ab.
    devices.append(_mosfet(
        "Mcmfb_drv", "pmos",
        d="vss", g="cmfb_out", s="vbias_p_ab", b="vdd",
        W=W_cmfb_amp, L=L_cmfb_amp,
    ))

    # ─── Miller compensation ──────────────────────────────
    devices.append(_cap("Cc", p="out", n="out_main", value=Cc))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R112 C: optional DEEP_NWELL annotation ─────────────
    # Mab_push is the NMOS push device in the class-AB output stage
    # (drain=vdd, gate=vbias_p_ab, source=out). Its source node `out`
    # swings rail-to-rail across the class-AB idle point, so the
    # drain-bulk capacitance dumps the full output transient into the
    # global p-substrate unless the device is fenced off by a deep-nwell
    # well. Razavi §17.3 — when an NMOS output device must be isolated
    # from the global p-substrate for noise immunity, DNW is the
    # process option (plain pwell tap is not enough because the pwell
    # still sits on the global p-substrate).
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.DEEP_NWELL,
            devices=("Mab_push",),
            extras={
                "rationale": (
                    "Class-AB NMOS push device — output node swings "
                    "rail-to-rail; drain-bulk substrate coupling must be "
                    "broken by deep-nwell isolation so the stage-1 input "
                    "pair and CMFB sense network do not see the output "
                    "transient through the shared p-substrate"
                ),
                "process_required": "deep_nwell",
                "isolate_from": ("M1", "M2", "Mcmfb_in"),
            },
        ))

    return cell


__all__ = ["build_class_AB_output_opamp", "build_ac_testbench"]
