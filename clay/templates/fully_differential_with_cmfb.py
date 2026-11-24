"""
Built-in template: fully_differential_opamp_with_cmfb.

Fully-differential single-stage opamp with a DEDICATED CMFB CONTROL
node (``vcmfb``) that drives the NMOS active-load gates through an
internal inverting stage. This is the v0.0.84 closure of the v0.0.82 /
v0.0.83 honest-scope deferred finding: the existing fully-diff
testbench's CMFB regulator output (the top-level ``vcmfb`` net driven
by ``E_cmfb``) had no DUT-side handle to drive — the loop closed only
through the 1 MΩ ``Rcm1``/``Rcm2`` sensor-leakage path. As a result,
``cmfb_cm_offset`` landed at a device-determined value (≈ −900 mV on
the v0.0.82 synthetic pair, regardless of ``cmfb_ea_gain``).

This template provides a DUT cell whose internal wiring closes the
loop on the testbench's ``vcmfb`` node, so that sweeping
``cmfb_ea_gain`` produces a monotonically increasing ``|cmfb_cm_offset|``
as gain drops — the operational definition of CMFB CM-regulation
degradation under finite EA gain.

Topology (single-stage FD, CMFB-controlled NMOS loads, sky130 1.8V)
====================================================================

                      vdd ─────────┬───────────┬─────────────┬────────
                                   │           │             │
                                  Mbp        Mtail (W=N_tail*Wbp)
                                   │           │             │
                                   │           tail          │
                                   │           │           Minv_p
                                   │       ┌───┴───┐         │
                                   │      Mdp_p  Mdp_n       │
                             pbias─┤       │       │         │
                                   │   inp─┤   inn─┤         │
                                   │       │       │         │
                                   │      outn    outp      ┌┴──── vcmfb_inv ──┐
                                   │       │       │        │                  │
                                   ├───────┘       └────────│──── Mload_p ───┐ │
                                   │                        │                │ │
                                   ├───────────────────────┐│   Mload_n ─────┘ │
                                   │                       ││                  │
                                   │                       └┴──── Minv_n ──── vcmfb (global)
                                   │                                │
                                  Iref (sinks to vss)                vss
                                   │
                                  vss

Bias plan
---------
- ``Iref`` (current source) sinks a small reference current from
  ``pbias`` to ``vss``. ``Mbp`` (PMOS diode, source=vdd, drain=gate=pbias)
  sources this current, settling ``pbias`` at roughly ``vdd − |Vgs_p|``.
- ``Mtail`` is a PMOS mirror of ``Mbp`` at ratio ``N_tail``, supplying
  ``I_tail ≈ N_tail · I_ref`` to the diff-pair sources.
- ``Mdp_p`` and ``Mdp_n`` form the PMOS input pair (sources tied to
  ``tail``, gates on ``inp``/``inn``, drains on ``outn``/``outp``).
  Polarity convention follows the v0.0.22 ``FullyDifferentialPMOSInput``:
  ``Mdp_p.gate=inp → drain=outn`` so a positive differential input
  ``Vinp − Vinn`` yields a positive ``Voutp − Voutn``.

CMFB-controlled load (key innovation)
-------------------------------------
The NMOS active loads ``Mload_p`` / ``Mload_n`` have their gates wired
to an internal node ``vcmfb_inv`` (NOT to ``vcmfb`` directly). This
internal node is the output of a CMOS inverting stage driven by
``vcmfb``:

    Minv_n: NMOS, gate=vcmfb,  source=vss, drain=vcmfb_inv
    Minv_p: PMOS, gate=pbias,  source=vdd, drain=vcmfb_inv

The inversion is mathematically necessary. The v0.0.77 testbench's
CMFB error amplifier emits ``vcmfb = gain · (vcm_target − vcm_sensed)``,
so when the output common mode is too LOW (``vcm_sensed < vcm_target``)
the regulator drives ``vcmfb`` HIGH. To CLOSE the loop with negative
feedback we need that condition to PULL OUTPUTS UP, which means
sinking LESS current through the NMOS loads — i.e., NMOS load gates
should DROP when ``vcmfb`` RISES. The internal common-source NMOS
inverter (``Minv_n`` + ``Minv_p`` pull-up) implements that inversion.

Walking the loop end-to-end:

  * Outputs too LOW → ``vcm_sensed`` < ``vcm_target`` → testbench
    drives ``vcmfb`` HIGH.
  * High ``vcmfb`` → ``Minv_n`` conducts more → ``vcmfb_inv`` DROPS.
  * Low ``vcmfb_inv`` → ``Mload_p``/``Mload_n`` Vgs smaller → less
    current sunk to ``vss`` → outputs RISE.

Negative feedback ✓. With a high-gain ideal EA (``cmfb_ea_gain=1e6``)
the loop drives the output common mode to within < 1 µV of
``vcm_target``. With finite EA gain the loop has a steady-state
residual that scales as roughly ``vcm_target / gain`` for a single-pole
EA topology. The quantitative .meas line ``cmfb_cm_offset`` introduced
in v0.0.82 reads this residual directly.

``.global vcmfb`` bridging requirement
--------------------------------------
The Clay 6-port fully-differential convention is
``[inn, inp, outn, outp, vdd, vss]`` — exactly the subset that
``_detect_port_shape`` routes to the v0.0.77 / 78 / 79 / 82 testbench
builders. To preserve that contract, this template does NOT expose
``vcmfb`` as a 7th external port; that would force a non-standard
xDUT instantiation line outside the existing FD testbench builders
(which the mission scope explicitly forbids touching).

Instead, ``vcmfb`` is a global net inside the exported subckt
(``simulate_with_cmfb_loop`` in ``clay.stdlib.fully_differential
.opamp_with_cmfb`` injects ``.global vcmfb`` immediately before
``.subckt`` in the exported netlist). ngspice promotes references
to ``vcmfb`` inside the subckt to the same top-level net the
testbench's ``E_cmfb`` writes — closing the CMFB loop through the
DUT without changing the testbench builders or the port-shape
detector.

Port list
---------
``[inn, inp, outn, outp, vdd, vss]`` — 6 ports. ``vcmfb`` is global,
not a port.

Sizing intent
-------------
- ``gm_id_diff``: applies to ``Mdp_p`` / ``Mdp_n`` (matched). ``I_d``
  ≈ ``I_tail/2`` per leg.
- ``gm_id_load``: applies to ``Mload_p`` / ``Mload_n`` (matched).
  Currents track the diff-pair currents under steady-state CMFB.
- The internal CMFB inverter (``Minv_n`` / ``Minv_p``) is small
  signal-path only — it carries a tiny bias current to set
  ``vcmfb_inv`` at the right OP for the loads.

Honest scope (v0.0.84)
----------------------
- This is a single-stage architecture (no gain boost, no output buffer).
  The differential gain is ``gm_dp · (r_o_dp ‖ r_o_load)`` ≈ 20–30 dB
  on sky130 with the default sizing.
- The CMFB inverter introduces a single intermediate pole. For the
  v0.0.81 phase-margin probe, the dominant pole is still the EA's
  internal pole; ``cmfb_phase_margin`` therefore continues to read
  the EA's open-loop response, not the closed-loop CMFB stability.
- CM-mode AC ``.meas`` of ``cmfb_cm_offset`` is a v0.0.82 layer's
  measure; on a working FD amp the AC builder's ``find v(outp) at=f``
  returns the AC RESPONSE at the probe frequency rather than the DC
  operating point, so the AC ``cmfb_cm_offset`` is dominated by the
  AC differential swing (which is balanced and cancels — leaving
  ``−vcm_target = −0.9 V`` as a constant). The transient and
  DC-sweep ``cmfb_cm_offset`` measures DO use a late-window AVG and
  read the steady-state CM residual correctly; the DC-sweep number
  is the canonical demonstration of the gain dependence.
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


def build_fully_diff_opamp_with_cmfb(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 2e-5,
    W_bp: float = 4e-6,
    L_bp: float = 1e-6,
    N_tail: int = 4,

    # ─── Diff pair sizing ───────────────────────────────────
    W_diff: float = 20e-6,
    L_diff: float = 0.5e-6,

    # ─── NMOS load sizing ───────────────────────────────────
    W_load: float = 4e-6,
    L_load: float = 1e-6,

    # ─── CMFB inverter sizing ───────────────────────────────
    # NMOS input of the inverter. Wider than the loads so its
    # transconductance dominates the inverter's loop gain.
    W_inv_n: float = 8e-6,
    L_inv_n: float = 0.5e-6,
    # PMOS pull-up biased from pbias (mirror leg).
    W_inv_p: float = 4e-6,
    L_inv_p: float = 1e-6,

    # R119 C: optional qualitative DRC annotations
    # When True, attach three R111 C-vocabulary annotations on the
    # fully-differential single-stage opamp with internal CMFB
    # inverter:
    #   * MATCHING on (Mdp_p, Mdp_n). PMOS diff pair — Vth mismatch
    #     becomes input-referred offset and breaks the cross-coupled
    #     polarity contract (Vinp-Vinn -> Voutp-Voutn) that makes the
    #     FD signal path differentially balanced.
    #   * SYMMETRIC_LAYOUT on (Mload_p, Mload_n). NMOS active-load
    #     pair both driven by vcmfb_inv. Asymmetric layout (one closer
    #     to the inverter output than the other) creates a parasitic
    #     RC mismatch on the vcmfb_inv -> gate path that translates
    #     CMFB regulation transients into a small differential output
    #     glitch, undermining the FD architecture's whole point.
    #   * GUARD_RING on Mtail (PMOS tail, M=N_tail). Carries the
    #     entire stage current N_tail * I_ref; nwell guard ring tied
    #     to vdd isolates the tail node from supply-coupled substrate
    #     noise. With CMFB closed, tail noise leaks to the regulated
    #     common-mode at a rate set by the EA's finite gain.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a fully-differential single-stage opamp
    with CMFB-controlled NMOS loads.

    The output common-mode is regulated by a top-level CMFB error
    amplifier (provided by the v0.0.77 fully-diff testbench family)
    that drives a global ``vcmfb`` net. This cell wires the loop into
    the DUT via an internal CMOS-inverter stage (``Minv_n`` /
    ``Minv_p``) so the testbench's polarity convention closes the
    loop with negative feedback.

    Args:
        cell_name: name for the Cell IR (becomes the .subckt name).
        spec: target spec (gain_dc, gbw, ...). Used by the higher-
            level register_cell path; pure sizing-only callers can
            pass an empty Spec.
        I_ref: reference current sunk from ``pbias`` to ``vss`` (A).
        W_bp, L_bp: PMOS bias-diode sizing (Mbp and Mtail).
        N_tail: PMOS tail-mirror ratio (Mtail/Mbp).
        W_diff, L_diff: PMOS diff-pair sizing (Mdp_p / Mdp_n matched).
        W_load, L_load: NMOS active-load sizing (Mload_p / Mload_n
            matched). Gates wired to ``vcmfb_inv``.
        W_inv_n, L_inv_n: CMFB-inverter NMOS sizing (gate on vcmfb).
        W_inv_p, L_inv_p: CMFB-inverter PMOS pull-up sizing (gate on
            pbias).

    Returns:
        A Cell IR with 6 external ports
        ``[inn, inp, outn, outp, vdd, vss]`` and a global ``vcmfb``
        net wired to the inverter input.

    Notes:
        Use ``clay.stdlib.fully_differential.opamp_with_cmfb
        .simulate_with_cmfb_loop(handle, ...)`` to run this cell
        through the v0.0.77 fully-diff testbench family. That wrapper
        injects the required ``.global vcmfb`` directive into the
        exported netlist; without it the testbench's top-level
        ``vcmfb`` net (driven by ``E_cmfb``) would not connect to the
        DUT's internal ``vcmfb`` references.
    """
    ports = ["inn", "inp", "outn", "outp", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Bias chain ──────────────────────────────────────────
    # Iref sinks I_ref from pbias to vss. With Mbp diode-connected
    # (g=d=pbias, s=vdd), the diode supplies that current and pbias
    # settles at roughly vdd - |Vgs_p|. This is the PMOS-bias seed.
    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "pbias", "n": "vss"},
        sizing={"value": I_ref},
    ))
    devices.append(_mosfet(
        "Mbp", "pmos", d="pbias", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp,
    ))

    # PMOS tail mirror: gate=pbias, source=vdd, drain=tail. Mirrors
    # Iref at ratio N_tail so I_tail ≈ N_tail · I_ref.
    devices.append(_mosfet(
        "Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
        W=W_bp, L=L_bp, M=N_tail,
    ))

    # ─── PMOS differential pair ──────────────────────────────
    # Polarity: Mdp_p.gate=inp drains to outn (cross-coupled with
    # Mdp_n.gate=inn → outp). Positive (Vinp - Vinn) → positive
    # (Voutp - Voutn).
    devices.append(_mosfet(
        "Mdp_p", "pmos", d="outn", g="inp", s="tail", b="tail",
        W=W_diff, L=L_diff,
    ))
    devices.append(_mosfet(
        "Mdp_n", "pmos", d="outp", g="inn", s="tail", b="tail",
        W=W_diff, L=L_diff,
    ))

    # ─── NMOS active loads driven by vcmfb_inv ──────────────
    # Gates wired to the inverter output (vcmfb_inv). Sources / bulks
    # at vss. The CMFB loop sets vcmfb_inv to whatever level makes
    # I_load match the per-leg diff-pair current at the chosen common-
    # mode operating point.
    devices.append(_mosfet(
        "Mload_p", "nmos", d="outp", g="vcmfb_inv", s="vss", b="vss",
        W=W_load, L=L_load,
    ))
    devices.append(_mosfet(
        "Mload_n", "nmos", d="outn", g="vcmfb_inv", s="vss", b="vss",
        W=W_load, L=L_load,
    ))

    # ─── CMFB inverter ──────────────────────────────────────
    # Common-source NMOS (Minv_n) with PMOS current-source pull-up
    # (Minv_p). Drives vcmfb_inv inverted relative to vcmfb. The
    # PMOS pull-up gate sits on pbias so it carries a fixed bias
    # current that sets the inverter's quiescent operating point.
    devices.append(_mosfet(
        "Minv_n", "nmos", d="vcmfb_inv", g="vcmfb", s="vss", b="vss",
        W=W_inv_n, L=L_inv_n,
    ))
    devices.append(_mosfet(
        "Minv_p", "pmos", d="vcmfb_inv", g="pbias", s="vdd", b="vdd",
        W=W_inv_p, L=L_inv_p,
    ))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R119 C: optional DRC annotations
    # MATCHING (diff pair) / SYMMETRIC_LAYOUT (NMOS loads) /
    # GUARD_RING (Mtail). See kwarg docs above for the per-annotation
    # rationale.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mdp_p", "Mdp_n"),
            extras={
                "rationale": (
                    "PMOS diff pair — Vth mismatch becomes input-"
                    "referred offset and breaks the cross-coupled "
                    "polarity contract (Vinp-Vinn -> Voutp-Voutn) "
                    "that keeps the FD signal path differentially "
                    "balanced"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.SYMMETRIC_LAYOUT,
            devices=("Mload_p", "Mload_n"),
            extras={
                "rationale": (
                    "NMOS active-load pair both driven by vcmfb_inv. "
                    "Asymmetric routing (one closer to the inverter "
                    "output than the other) creates a parasitic RC "
                    "mismatch on the vcmfb_inv -> gate path that "
                    "translates CMFB regulation transients into a "
                    "small differential output glitch, undermining "
                    "the FD architecture's whole point"
                ),
                "axis": "vertical",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "PMOS tail current source (M=N_tail) carries the "
                    "entire stage current N_tail * I_ref; nwell guard "
                    "ring tied to vdd isolates the tail node from "
                    "supply-coupled substrate noise. With CMFB closed, "
                    "tail noise leaks to the regulated common-mode at "
                    "a rate set by the EA's finite gain"
                ),
                "well_type": "nwell",
            },
        ))

    return cell
