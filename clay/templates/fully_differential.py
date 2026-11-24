"""
Built-in template: fully_differential_pmos_input.

Single-stage fully-differential amplifier (FDA) with a PMOS input pair
and an NMOS active-load pair. First Clay stage template with TWO
output ports (`out_p`, `out_n`) — essential for high-SNR ADC drivers,
sigma-delta loops, and modern data converters.

Topology (single-stage FDA, replica-bias load, sky130 1.8V)
============================================================

           vdd ─────────┬──────────────┬────────────────
                        │              │
                       Mbp         Mtail (W = N_tail · Wbp)
                        │              │
                        ├──┐           │
                        │  │           tail
                       Iref│           │
                        │  └───── pbias│
                        │              │
                        │           ┌──┴────┬─────┐
                        │          M1     M2     Msense   ← Three PMOS
                        │           │     │       │         each draws
                  inp ──┤   inn ────┤     │       │ ← vcm_int
                        │           │     │       │
                        │         out_n  out_p   nbias_load
                        │           │     │       │
                        │          M3    M4    Mbn_load(diode)
                        │           │     │       │
                        │          vss   vss     vss
                        │
                  (replica current sense)
                       vcm_int = (inp + inn) / 2  (via two large R)

Bias plan
---------
- Iref is injected internally into nbias (NMOS bias seed). NMOS diode
  Mbn absorbs it; mirror Mbnm pulls Iref from pbias to vss; PMOS diode
  Mbp at pbias supplies it from vdd. pbias settles ≈ vdd − |Vgs_pmos|.
- Mtail is a PMOS mirror of Mbp at ratio N_tail, providing the diff-pair
  tail current I_tail ≈ N_tail · I_ref (constrained by saturation).

Replica-bias for NMOS load (key innovation of this template)
------------------------------------------------------------
A naive FDA biases the NMOS loads M3/M4 from a separate reference current
chain. The closed-form gm/Id solver predicts certain currents, but sky130
+ VCM=mid-rail forces the PMOS diff pair into sub-Vth where actual I_d
is much smaller than the closed-form predicts. Result: the load mirror
demands ~40 µA per leg, the diff pair only delivers a few µA, the outputs
slam to vss within ngspice's first DC solve. (Folded-cascode and
telescopic-cascode templates work because their cascodes maintain high
output impedance even when currents are mismatched.)

Solution: REPLICA-BIAS. We add a third PMOS device `Msense` that
duplicates M1/M2's bias conditions exactly:
    Msense.size = M1.size = M2.size
    Msense.source = tail
    Msense.gate   = vcm_int   (= (V_inp + V_inn) / 2)
    Msense.drain  = nbias_load
A small resistive network produces `vcm_int` as the input common-mode
average. Msense's drain current is therefore identical to the per-leg
diff-pair current, regardless of whether M1/M2 operate in strong inversion
or sub-Vth. An NMOS diode `Mbn_load` at nbias_load absorbs Msense's
current and sets the load gate voltage so M3/M4 mirror it exactly.

Net effect: M3/M4 each sink the same current that M1/M2 each source,
to first order. The outputs settle at a sensible common-mode (typically
~0.5–0.6 V on sky130 1V8 at VCM=0.9 V) with all six transistors in
saturation. AC gain is gm_M1 · (r_o_M1 ‖ r_o_M3).

PMOS-input polarity convention
------------------------------
For positive differential gain
    Vout_p − Vout_n = Av · (Vinp − Vinn),
M1's drain must be wired to out_n and M2's to out_p:
    Vinp ↑ → |Vgs_M1| ↓ → I_M1 ↓ → out_n ↓
    Vinn ↑ → |Vgs_M2| ↓ → I_M2 ↓ → out_p ↑  (when Vinn drops)
Symmetry: when Vinp − Vinn rises, out_p − out_n rises. Correct sign.

Soft common-mode pinning (defense-in-depth)
-------------------------------------------
On top of the replica-bias scheme, a pair of very-high-impedance
resistors (default 100 MΩ each) ties out_p and out_n to an internally
generated mid-rail node `vcm_ref`. This is purely defensive — the
replica bias already locks the OP — but it prevents the common-mode
from drifting at DC under model corner shifts. AC gain is barely
affected because 100 MΩ ‖ r_o ≈ r_o.

Port list
---------
['inn', 'inp', 'out_p', 'out_n', 'vdd', 'vss'] — 6 ports. The two output
ports carry the differential signal.

Sizing intent
-------------
- `gm_id_diff`: applies to M1 / M2 / Msense (all three matched). I_d ≈
  I_tail/2 in the closed-form sense; actual I_d is set by sky130 OP.
- `gm_id_load`: applies to M3 / M4 / Mbn_load. I_d nominally = I_tail/2;
  via the replica, M3/M4 currents track M1/M2 currents.

Known limitations (v0.0.22)
---------------------------
- No true CMFB loop. The replica-bias scheme locks the differential-pair
  current but doesn't actively reject common-mode disturbances at the
  output beyond what the soft tie-down resistors provide. Real switched-
  cap or continuous CMFB is a follow-up milestone.
- The replica adds one extra device (Msense) and three resistors (two
  input-sense + the soft tie-down) compared with a naive FDA. Acceptable
  cost given the OP-convergence benefit.
"""

from __future__ import annotations

from ..ir import (
    AcAnalysis, Cell, Device, Measure, OpAnalysis, Source, Spec, Testbench,
)


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


def build_fully_differential_pmos_input(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 2e-5,
    W_bp: float = 4e-6, L_bp: float = 1e-6,
    N_tail: int = 4,

    # ─── Per-device sizing: physical (W, L) form ────────────
    # Used unless overridden by `gm_id_*` intent (next section).
    W_diff: float = 20e-6, L_diff: float = 0.5e-6,
    W_load: float = 8e-6,  L_load: float = 1e-6,

    # ─── Per-device sizing: intent (gm/Id) form ─────────────
    # When provided, derives W from (gm_id, I_d, L) via the closed-form
    # solver. I_d for each device is derived from the bias plan.
    gm_id_diff: float | None = None,
    gm_id_load: float | None = None,

    # ─── Replica-bias common-mode sense ─────────────────────
    R_vcm_sense: float = 10e6,    # inp↔vcm_int and inn↔vcm_int

    # ─── Soft CMFB anchor (defense-in-depth) ────────────────
    R_cmfb: float = 100e6,        # out_p/out_n ↔ vcm_ref
    R_div: float = 10e6,          # vdd↔vcm_ref and vcm_ref↔vss

    # CL is kept here for forward-compat with sizing-from-spec; the
    # testbench builder wires the actual differential load caps.
    CL: float = 10e-12,

    # ─── R112 C: optional qualitative DRC annotations ───────
    # When True, attach a SYMMETRIC_LAYOUT annotation on the differential
    # signal-path group (M1 / M2 PMOS input pair + M3 / M4 NMOS load pair).
    # The fully-differential topology is gain-element-symmetry limited
    # for CMRR / PSRR — common-centroid alone is not enough; the layout
    # must respect a mirror axis with M1↔M2 on opposite sides and
    # M3↔M4 on opposite sides, both axis-symmetric to the tail node.
    # The new R111 C SYMMETRIC_LAYOUT kind carries the axis orientation
    # and anchor net explicitly (axis=vertical, anchor=tail) so the
    # layout prompt knows the symmetry constraint geometry. Annotations
    # live only in cell.__dict__["_drc_annotations"]; SPICE emit (which
    # iterates cell.devices) is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a single-stage fully-differential amplifier
    with PMOS input pair and NMOS active loads on sky130.

    Sizing inputs come in TWO equivalent forms — pick one per device:
      * Explicit W / L kwargs (legacy)
      * Intent kwargs (`gm_id_*`): the closed-form gm/Id solver in
        `clay.sizing` derives W from design intent. This is the
        LLM-native interface — LLM writes gm/Id targets, the solver
        produces concrete W values.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (gain_dc, gbw, ...).
        I_ref: reference current into the bias chain.
        W_bp, L_bp: bias-mirror PMOS sizing.
        N_tail: PMOS mirror ratio Mtail/Mbp.
        W_diff, L_diff: M1 / M2 / Msense PMOS diff-pair sizing (matched).
        W_load, L_load: M3 / M4 / Mbn_load NMOS active-load sizing (matched).
        gm_id_diff: design intent for the diff pair and replica sense.
        gm_id_load: design intent for the NMOS loads and load diode.
        R_vcm_sense: input-common-mode sense resistors (10 MΩ default).
        R_cmfb: defensive soft tie-down to mid-rail (100 MΩ default).
        R_div: vcm_ref divider leg (10 MΩ default).
        CL: forward-compat load capacitance; testbench owns the actual cap.

    Returns:
        A Cell IR with 6 ports: ['inn', 'inp', 'out_p', 'out_n', 'vdd', 'vss'].
    """
    from ..sizing import closed_form_W

    # Per-device drain currents under the bias plan.
    I_diff = (N_tail * I_ref) / 2.0   # each diff-pair leg
    I_load = I_diff                   # each load mirrors I_diff

    if gm_id_diff is not None:
        W_diff = closed_form_W(
            "pmos", gm_id=gm_id_diff, I_d=I_diff, L=L_diff,
        ).W
    if gm_id_load is not None:
        W_load = closed_form_W(
            "nmos", gm_id=gm_id_load, I_d=I_load, L=L_load,
        ).W

    ports = ["inn", "inp", "out_p", "out_n", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Bias chain ──────────────────────────────────────────
    devices.append(_mosfet("Mbp",  "pmos", d="pbias", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbn",  "nmos", d="nbias", g="nbias", s="vss", b="vss",
                            W=W_bp, L=L_bp))
    devices.append(_mosfet("Mbnm", "nmos", d="pbias", g="nbias", s="vss", b="vss",
                            W=W_bp, L=L_bp))

    # Diff-pair tail current source (PMOS mirror of Mbp, ratio N_tail).
    devices.append(_mosfet("Mtail", "pmos", d="tail", g="pbias", s="vdd", b="vdd",
                            W=W_bp, L=L_bp, M=N_tail))

    # ─── PMOS differential pair ──────────────────────────────
    # M1: gate=inp, drain=out_n   (PMOS-input polarity convention)
    # M2: gate=inn, drain=out_p
    devices.append(_mosfet("M1", "pmos", d="out_n", g="inp", s="tail", b="vdd",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos", d="out_p", g="inn", s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # ─── Replica-bias common-mode sense ──────────────────────
    # vcm_int = (V_inp + V_inn) / 2 via a resistive averager.
    # Msense duplicates M1/M2 bias exactly so its drain current equals
    # I_diff (per leg) regardless of sub-Vth / strong-inversion regime.
    devices.append(_res("R_vcm_p", p="inp", n="vcm_int", value=R_vcm_sense))
    devices.append(_res("R_vcm_n", p="inn", n="vcm_int", value=R_vcm_sense))
    devices.append(_mosfet("Msense", "pmos", d="nbias_load", g="vcm_int",
                            s="tail", b="vdd",
                            W=W_diff, L=L_diff))

    # ─── NMOS active loads ───────────────────────────────────
    # Mbn_load is diode-connected at nbias_load, absorbing Msense's
    # current. M3/M4 mirror Mbn_load at the same W/L → each sinks
    # I_diff per leg, matching M1/M2.
    devices.append(_mosfet("Mbn_load", "nmos", d="nbias_load", g="nbias_load",
                            s="vss", b="vss",
                            W=W_load, L=L_load))
    devices.append(_mosfet("M3", "nmos", d="out_n", g="nbias_load", s="vss", b="vss",
                            W=W_load, L=L_load))
    devices.append(_mosfet("M4", "nmos", d="out_p", g="nbias_load", s="vss", b="vss",
                            W=W_load, L=L_load))

    # ─── Internal Iref source ────────────────────────────────
    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    # ─── Soft CMFB anchor (defense-in-depth) ─────────────────
    devices.append(_res("R_div_top", p="vdd",     n="vcm_ref", value=R_div))
    devices.append(_res("R_div_bot", p="vcm_ref", n="vss",     value=R_div))
    devices.append(_res("R_cmfb_p",  p="out_p",   n="vcm_ref", value=R_cmfb))
    devices.append(_res("R_cmfb_n",  p="out_n",   n="vcm_ref", value=R_cmfb))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R112 C: optional SYMMETRIC_LAYOUT annotation ───────
    # The fully-differential signal path crosses 4 matched gain elements
    # (M1 / M2 PMOS input pair + M3 / M4 NMOS load pair). The tail node
    # is the anchor: V(tail) is the common-mode reference for the input
    # pair, and the diff-pair drains (out_n / out_p) symmetrically
    # straddle the mirror axis. Razavi §11.3 — for CMRR / PSRR the
    # gain-element symmetry must hold at the LAYOUT level, not just the
    # device level. Common-centroid is necessary but not sufficient —
    # the layout must place M1 on the axis left, M2 on the axis right,
    # M3 on the axis left below M1, M4 on the axis right below M2, so
    # the differential drain currents see identical parasitic environments.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.SYMMETRIC_LAYOUT,
            devices=("M1", "M2", "M3", "M4"),
            extras={
                "rationale": (
                    "Fully-differential gain-element symmetry — M1/M2 PMOS "
                    "input pair and M3/M4 NMOS load pair must sit on the "
                    "same vertical mirror axis anchored at the tail node so "
                    "out_n / out_p see identical parasitic environments"
                ),
                "axis": "vertical",
                "anchor": "tail",
            },
        ))

    return cell


# ─── Testbench builder for AC characterization ─────────────────────

def build_ac_testbench_fda(
    cell: Cell,
    supply: float | None = None,
    vcm_ratio: float = 0.5,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_dec: int = 20,
    CL: float = 5e-12,
) -> Testbench:
    """AC testbench for a single-stage fully-differential amplifier.

    Drives a standard differential input via two VCVS sources around a
    common-mode reference; loads both outputs with matching CL to vss.

    Differential drive
    ------------------
    A single AC source VSIG drives net `sig`. Edp produces inp ≈ vcm + sig/2;
    Edn produces inn ≈ vcm − sig/2. So the differential input swing is
    (inp − inn) = sig, exactly as in the existing diff-opamp testbenches.

    Single-ended-equivalent measurement
    -----------------------------------
    The Clay measure_expr grammar does not (yet) support arithmetic like
    `vdb(out_p) - vdb(out_n)`. For v0.0.22 we measure only `out_p`'s
    response to the differential input drive — the "single-ended-
    equivalent view" of the FDA.

    Because the topology is fully symmetric, |v_out_p| ≈ |v_out_n| at
    all frequencies, so the true differential gain in dB is
    approximately the reported gain_dc + 6 dB. Users can derive the
    full differential gain by this 6 dB correction. This convention is
    documented in the FDA demo .clay file.

    `supply=None` auto-resolves from `cell.pdk_id`.
    """
    if supply is None:
        from ._supply import resolve_supply
        supply = resolve_supply(cell)
    sources: list[Source] = [
        Source("voltage_dc", "VDD",
               pins={"p": "vdd", "n": "0"},
               params={"dc": supply}),
        Source("voltage_dc", "VSS",
               pins={"p": "vss", "n": "0"},
               params={"dc": 0.0}),
        Source("voltage_dc", "VCM",
               pins={"p": "vcm", "n": "0"},
               params={"dc": supply * vcm_ratio}),
        Source("voltage_ac", "VSIG",
               pins={"p": "sig", "n": "0"},
               params={"dc": 0.0, "ac": 1.0}),
    ]
    # Differential VCVS pair: convert sig into ±sig/2 around vcm at inp/inn.
    extras = [
        Device("Edp", "vcvs", None,
               pins={"op": "inp", "on": "vcm", "ip": "sig", "in": "0"},
               sizing={"gain": 0.5}),
        Device("Edn", "vcvs", None,
               pins={"op": "inn", "on": "vcm", "ip": "sig", "in": "0"},
               sizing={"gain": -0.5}),
        # Symmetric load caps on both outputs.
        Device("CL_p", "cap", None,
               pins={"p": "out_p", "n": "vss"},
               sizing={"value": CL}),
        Device("CL_n", "cap", None,
               pins={"p": "out_n", "n": "vss"},
               sizing={"value": CL}),
    ]
    measures = [
        # Single-ended-equivalent view: out_p only.
        # True differential gain in dB ≈ this value + 6 dB.
        Measure("gain_dc",      "ac", "vdb(out_p) @ freq=1"),
        Measure("gbw",          "ac", "freq when vdb(out_p)=0"),
        Measure("phase_at_gbw", "ac", "vp(out_p) when vdb(out_p)=0"),
    ]
    analyses: list = [
        OpAnalysis(),
        AcAnalysis(sweep_kind="dec", points_per_decade=points_per_dec,
                   f_start=f_start, f_stop=f_stop),
    ]
    return Testbench(
        cell=cell,
        sources=sources,
        extra_devices=extras,
        analyses=analyses,
        measures=measures,
        pdk_corner="tt",
        supply_volt=supply,
    )
