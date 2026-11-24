"""
Built-in template: common_source_pmos_input.

Single-stage, single-ended common-source amplifier with PMOS input and
NMOS active-load current sink. The simplest practical amplifier topology
and the first non-differential stage template in Clay's standard library.

Topology
========

           vdd ────────┬────────────────
                       │
                      M_in (PMOS, gate=in, drain=out)
                       │
                       out ───────────► (output)
                       │
                      M_load (NMOS, gate=nbias, source=vss)
                       │
                      vss

Bias chain (sets nbias)
-----------------------
   vdd ─ [Iref] ─ nbias ─ Mbn(diode, NMOS) ─ vss
   M_load mirrors Mbn at ratio N_load, so I_d(M_load) ≈ N_load · I_ref.

Port list: (in, out, vdd, vss).
No differential inputs — this is a single-ended amplifier. The testbench
drives `in` directly with a midrail DC bias overlaid with a small AC
perturbation.

Sizing intent
-------------
A single `gm_id_input` kwarg overrides the explicit `W_gain` value via
the closed-form gm/Id solver. Typical CS amp design targets gm/Id ≈ 15-20
for low-noise input devices, derived from the bias current that the
NMOS sink imposes on the gain transistor (the PMOS self-biases to that
current at whatever Vgs is required given its sizing).

I_d for the gain transistor under the bias plan:
    I_gain = N_load * I_ref
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


def build_common_source_pmos_input(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 2e-5,
    W_bn: float = 4e-6, L_bn: float = 1e-6,
    N_load: int = 4,

    # ─── Per-device sizing: physical (W, L) form ────────────
    # Used unless overridden by gm_id_input (next).
    W_gain: float = 4e-5, L_gain: float = 5e-7,

    # ─── Per-device sizing: intent (gm/Id) form ─────────────
    # When provided, derives W_gain from (gm_id, I_d, L) via
    # clay.sizing.closed_form_W. I_d is set by the bias plan
    # (N_load * I_ref).
    gm_id_input: float | None = None,

    CL: float = 10e-12,

    # R119 C: optional qualitative DRC annotations
    # When True, attach three R111 C-vocabulary annotations on the
    # single-stage common-source PMOS-input amp:
    #   * GUARD_RING on Min (PMOS input/gain device). Carries the
    #     entire stage current N_load * I_ref; nwell guard ring tied
    #     to vdd isolates the high-Z `out` node from supply-coupled
    #     substrate noise. Single-ended CS amps have no diff-pair
    #     cancellation, so substrate noise couples 1:1 to out.
    #   * CURRENT_MIRROR_MATCHING on (Mbn, Mload). Bias-diode + active-
    #     load mirror at ratio N_load = Mload.M / Mbn.M. Mismatch in
    #     Vth or W/L between Mbn and Mload skews I_gain away from the
    #     designed N_load * I_ref, shifting the OP and the gm of the
    #     PMOS input — i.e., directly skewing gain_dc.
    #   * SYMMETRIC_LAYOUT on (Mbn, Mload). Bias-diode + mirror gate-
    #     line common to nbias; equal trace lengths from nbias to both
    #     gate terminals preserve the mirror ratio across the layout.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the single-stage common-source PMOS-input amp.

    Sizing inputs come in TWO equivalent forms — pick one:
      * Explicit W_gain / L_gain (legacy)
      * Intent `gm_id_input`: the closed-form gm/Id solver in
        `clay.sizing` derives W_gain from design intent.

    The bias-chain device sizing (W_bn, L_bn) and N_load mirror ratio
    fix the gain transistor's drain current:
        I_gain = N_load * I_ref

    The PMOS input self-biases to whatever Vgs is needed to carry that
    current at the chosen W/L.

    Args:
        cell_name: name for the Cell IR
        spec: target spec (gain_dc, gbw, ...)
        I_ref: reference current into the NMOS bias diode
        W_bn, L_bn: bias-diode width/length
        N_load: mirror ratio M_load.W / M_bn.W; sets I_gain = N_load*I_ref
        W_gain, L_gain: PMOS input transistor sizing (overridden by
            `gm_id_input` if provided)
        gm_id_input: design intent for the gain transistor; W is derived
        CL: load capacitance pin in the testbench (not used by the cell
            itself, kept here for forward compat with sizing-from-spec)
    """
    # Lazy import to avoid hard dep cycle.
    from ..sizing import closed_form_W

    # Drain current of the gain transistor under the bias plan
    I_gain = N_load * I_ref

    if gm_id_input is not None:
        W_gain = closed_form_W(
            "pmos", gm_id=gm_id_input, I_d=I_gain, L=L_gain,
        ).W

    ports = ["in", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain: NMOS diode at nbias, current source pulls I_ref into nbias
    # from vdd so that the diode absorbs it to vss. nbias settles at Vgs_n.
    devices.append(_mosfet("Mbn", "nmos", d="nbias", g="nbias",
                            s="vss", b="vss",
                            W=W_bn, L=L_bn))

    # NMOS active-load current sink — mirror of Mbn at ratio N_load.
    # Carries I_gain = N_load * I_ref to vss.
    devices.append(_mosfet("Mload", "nmos", d="out", g="nbias",
                            s="vss", b="vss",
                            W=W_bn, L=L_bn, M=N_load))

    # PMOS input (gain) transistor: source=vdd, gate=in, drain=out
    devices.append(_mosfet("Min", "pmos", d="out", g="in",
                            s="vdd", b="vdd",
                            W=W_gain, L=L_gain))

    # Inject the reference current INTO nbias from vdd: the diode then
    # absorbs it to vss. Convention: cur_src p=vdd n=nbias, value=I_ref.
    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias"},
        sizing={"value": I_ref},
    ))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # R119 C: optional DRC annotations
    # GUARD_RING (Min PMOS) / CURRENT_MIRROR_MATCHING (Mbn, Mload) /
    # SYMMETRIC_LAYOUT (Mbn, Mload). See kwarg docs above for the
    # per-annotation rationale.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Min",),
            extras={
                "rationale": (
                    "PMOS input/gain device carries the entire stage "
                    "current N_load * I_ref; nwell guard ring tied to "
                    "vdd isolates the high-Z `out` node from supply-"
                    "coupled substrate noise. Single-ended CS amps "
                    "have no diff-pair cancellation, so substrate "
                    "noise couples 1:1 to out"
                ),
                "well_type": "nwell",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("Mbn", "Mload"),
            extras={
                "rationale": (
                    "Bias-diode (Mbn) + active-load mirror (Mload) at "
                    "ratio N_load = Mload.M / Mbn.M. Vth or W/L mismatch "
                    "between Mbn and Mload skews I_gain away from the "
                    "designed N_load * I_ref, shifting the OP and the "
                    "gm of the PMOS input — directly skewing gain_dc"
                ),
                "ratio": f"1:{N_load}",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.SYMMETRIC_LAYOUT,
            devices=("Mbn", "Mload"),
            extras={
                "rationale": (
                    "Bias-diode + mirror gate-line common to nbias; "
                    "equal trace lengths from nbias to both gate "
                    "terminals preserve the mirror ratio across the "
                    "layout — gate-line IR drop asymmetry would skew "
                    "the effective Vgs at the mirror output"
                ),
                "axis": "horizontal",
            },
        ))

    return cell


# ─── Testbench builder for AC characterization ─────────────────────

def build_ac_testbench_cs(
    cell: Cell,
    supply: float | None = None,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    points_per_dec: int = 20,
    CL: float = 10e-12,
    R_fb: float = 1e9,
    C_ac: float = 1.0,
) -> Testbench:
    """AC testbench for a single-ended common-source amplifier.

    Self-biasing test rig (the standard textbook CS amp characterization):

        Vac ── Cac ───┬─── in ───┬─── Rfb ───┐
                                                  │
                                              out (= DUT output)

    At DC, Cac is open so `in` is pulled to `out` through the large
    feedback resistor R_fb (≈ 1 GΩ). The CS amp self-biases to whatever
    Vgs it needs for the PMOS to carry the NMOS sink's I_load — no need
    for the user (or the LLM) to compute Vincm by hand for each sizing
    / corner / spec combination. This is the LLM-native default: the
    designer writes intent (gm_id_input = 15), and the testbench finds
    the operating point automatically.

    At AC (above 1/(2π·Rfb·Cac·gain) ≈ a few mHz), Cac is a short and
    Rfb is effectively open — `in` follows Vac directly. The .meas
    expressions read vdb(out) just like the differential testbench, so
    spec assertions on gain_dc / gbw / phase_at_gbw are interchangeable.

    The DUT itself sees the same single port `in` carrying both DC bias
    and AC perturbation — no testbench detail leaks into the cell.

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
        # AC source driving `sig` net (NOT `in` directly). Through Cac,
        # AC perturbation reaches `in`; DC is blocked.
        Source("voltage_ac", "VSIG",
               pins={"p": "sig", "n": "0"},
               params={"dc": 0.0, "ac": 1.0}),
    ]
    extras = [
        # AC coupling cap from sig to in
        Device("Cac", "cap", None,
               pins={"p": "sig", "n": "in"},
               sizing={"value": C_ac}),
        # DC feedback resistor: in ↔ out. At DC, ties in to out (self-bias).
        # At AC, R_fb is effectively open relative to typical amp output
        # impedance, so it does not load the small-signal response.
        Device("Rfb", "res", None,
               pins={"p": "out", "n": "in"},
               sizing={"value": R_fb}),
        # Load cap from out to vss
        Device("CL", "cap", None,
               pins={"p": "out", "n": "vss"},
               sizing={"value": CL}),
    ]
    measures = [
        Measure("gain_dc",      "ac", "vdb(out) @ freq=1"),
        Measure("gbw",          "ac", "freq when vdb(out)=0"),
        Measure("phase_at_gbw", "ac", "vp(out) when vdb(out)=0"),
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
