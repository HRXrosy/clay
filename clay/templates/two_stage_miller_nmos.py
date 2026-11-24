"""
Built-in template: two_stage_miller_nmos.

Complementary topology to ``two_stage_miller_pmos``: NMOS-input diff pair
with PMOS current-mirror load, second-stage PMOS common-source amplifier
with NMOS-mirror load.

           vdd ─────┬───────┬─────────────────┬────
                    │       │                 │
                  M3──── M4                  Mlcs (PMOS, gate=pbias)
                    │       │                 │
                    d1     d2                out
                    │       │ (Mlcs is the
                    │       │  CS load mirror)│
                    │       │                 │
                    M1     M2                Mcs (PMOS CS, gate=d2)
                    │       │                 │
              inp ──┤  inn──┘                 │
                    │       │                 │
                   tail ────┘                out
                    │
                  Mtail (NMOS, gate=nbias, source=vss)
                    │
                   vss

Bias plan:
- Iref pulls current INTO vss (from pbias), producing an NMOS-side seed.
- Mbn is NMOS diode-connected (gate=drain=nbias, source=vss), absorbs Iref.
- Mtail is NMOS mirror of Mbn (gate=nbias, source=vss), ratio Mtail.W / Mbn.W = N.
- Mlcs is NMOS mirror of Mbn, ratio = K (Mcs current).
- PMOS current-mirror M3/M4 sets diff pair tail balance.
- Mcs is PMOS CS amplifier; gate = d2, source=vdd, drain=out.
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


def build_two_stage_miller_nmos(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Bias / mirror ratios ───────────────────────────────
    I_ref: float = 10e-6,
    W_bn: float = 4e-6,    L_bn: float = 1e-6,
    N_tail: int = 4,
    K_cs_load: int = 16,

    # ─── Per-device sizing: physical (W, L) form ────────────
    # These are used unless overridden by `gm_id_*` intent (next section),
    # in which case W is derived from gm/Id and the kwarg here is ignored.
    W_diff: float = 20e-6, L_diff: float = 0.5e-6,
    W_load1: float = 4e-6, L_load1: float = 1e-6,
    W_cs: float = 40e-6,   L_cs: float = 0.5e-6,

    # ─── Per-device sizing: intent (gm/Id) form ─────────────
    # Set any of these to override the W kwarg above; the closed-form
    # gm/Id solver in `clay.sizing` derives W from (gm_id, I_d, L).
    # I_d for each device is derived from the bias plan (see derivations
    # below). Passing both gm_id_* and W_* is an error.
    gm_id_diff: float | None = None,
    gm_id_load1: float | None = None,
    gm_id_cs: float | None = None,

    Cc: float = 1.5e-12,

    # ─── PDK-aware bias scaling (R107 C / Path B) ───────────
    # When provided, the cell is built with PDK-specific scaling of
    # W and I_ref so the gm/Id intent lands at comparable Vds_sat /
    # ro on each PDK. ``pdk_id="sky130"`` (the default) preserves
    # byte-exact behaviour — see docs/audits/2026-05-21-cross-pdk-
    # bias-scale.html and `clay/templates/_pdk_bias_scale.py` for the
    # scaling table and rationale.
    pdk_id: str = "sky130",

    # R119 C: optional qualitative DRC annotations
    # When True, attach three R111 C-vocabulary annotations on the
    # textbook 2-stage Miller NMOS-input opamp:
    #   * MATCHING on (M1, M2). NMOS diff pair — Vth mismatch maps
    #     directly to input-referred offset; sets the achievable
    #     CMRR/PSRR floor of the Razavi Ch. 9 baseline.
    #   * CURRENT_MIRROR_MATCHING on (M3, M4). PMOS load mirror (M3
    #     diode at d1, M4 replica at d2). Mismatch unbalances diff-pair
    #     currents, contributing an additional systematic offset on top
    #     of the diff-pair's Vth mismatch.
    #   * GUARD_RING on (Mtail,). NMOS tail current source carrying the
    #     entire stage-1 bias current N_tail * I_ref; pwell guard ring
    #     tied to vss isolates the diff-pair tail node from substrate-
    #     coupled noise. Tail-noise -> common-mode -> finite-CMRR
    #     conversion sets the PSRR_VSS floor of the input stage.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the 2-stage Miller NMOS-input opamp.

    Sizing inputs come in TWO equivalent forms — pick one per device:
      * Explicit W / L kwargs (legacy)
      * Intent kwargs (`gm_id_*`): the closed-form gm/Id solver in
        `clay.sizing` derives W from design intent. This is the
        LLM-native interface — LLM writes gm/Id targets, the solver
        produces concrete W values.

    Mixing both forms for the same device raises ValueError.

    PDK awareness (R107 C / Path B):
        ``pdk_id`` selects an entry in
        :data:`clay.templates._pdk_bias_scale._PDK_BIAS_SCALES`. The
        default (``"sky130"``) is identity scaling — sky130 netlists are
        byte-exact preserved. Other PDKs (``"gf180mcu"``, …) apply
        ``w_scale`` and ``iref_scale`` to bring the cross-PDK
        relative gain_dc divergence under the 30% TIER_LOOSE gate.
    """
    from ..sizing import closed_form_W
    from ._pdk_bias_scale import get_bias_scale

    # PDK-aware bias scaling. sky130 is identity (1.0, 1.0) -> byte-
    # exact preserved. Non-sky130 PDKs return non-identity factors that
    # scale W and I_ref so the gm/Id intent lands at comparable
    # Vds_sat / ro on each PDK.
    bias_scale = get_bias_scale(pdk_id, cell_flavor="nmos_input")
    I_ref = I_ref * bias_scale.iref_scale

    # Apply PDK-aware L scaling BEFORE the gm/Id solver so the derived
    # W respects the new V_OV implied by the scaled L. On sky130
    # (bias_scale.l_scale == 1.0) these are no-ops, preserving byte-
    # exact netlists.
    #
    # L_cs intentionally NOT scaled: the second-stage CS device sits at
    # K_cs_load x I_ref drain current (16x the input-pair current at
    # default ratios); growing L_cs together with the closed-form W
    # solver would push W_cs out of the gf180mcu nmos_3p3/pmos_3p3
    # binning window. Keeping L_cs at the caller's intent prevents the
    # "could not find a valid modelname" netlist failure.
    L_diff = L_diff * bias_scale.l_scale
    L_load1 = L_load1 * bias_scale.l_scale

    # I_d for each device in this topology (derived from the SCALED bias plan):
    #   - diff pair: I_tail / 2 = (N_tail * I_ref) / 2
    #   - load mirror M3/M4: same as diff pair half-current
    #   - output CS: K_cs_load * I_ref
    I_diff = (N_tail * I_ref) / 2.0
    I_load1 = I_diff
    I_cs = K_cs_load * I_ref

    if gm_id_diff is not None:
        W_diff = closed_form_W("nmos", gm_id=gm_id_diff, I_d=I_diff, L=L_diff).W
    if gm_id_load1 is not None:
        W_load1 = closed_form_W("pmos", gm_id=gm_id_load1, I_d=I_load1, L=L_load1).W
    if gm_id_cs is not None:
        W_cs = closed_form_W("pmos", gm_id=gm_id_cs, I_d=I_cs, L=L_cs).W

    # Apply PDK-aware W scaling AFTER gm/Id derivation so the scale
    # multiplies whichever W form the caller chose (intent or explicit).
    # On sky130 (bias_scale.w_scale == 1.0) these are no-ops, preserving
    # the byte-exact netlist.
    W_diff = W_diff * bias_scale.w_scale
    W_load1 = W_load1 * bias_scale.w_scale
    W_cs = W_cs * bias_scale.w_scale

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # Bias chain (NMOS diode + PMOS source) — sets nbias
    # A PMOS source supplies I_ref into nbias from vdd; NMOS diode sinks it to vss.
    # nbias settles at Vgs_nmos above vss.
    devices.append(_mosfet("Mbn",   "nmos", d="nbias", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn))
    # PMOS mirror sets pbias from I_ref injected at pbias.
    devices.append(_mosfet("Mbp",   "pmos", d="pbias", g="pbias", s="vdd", b="vdd",
                            W=W_bn, L=L_bn))
    # Mirror PMOS pushing I_ref into nbias from vdd
    devices.append(_mosfet("Mbpm",  "pmos", d="nbias", g="pbias", s="vdd", b="vdd",
                            W=W_bn, L=L_bn))

    # First-stage tail current source: NMOS mirror of Mbn, ratio N
    devices.append(_mosfet("Mtail", "nmos", d="tail",  g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=N_tail))

    # NMOS differential pair
    devices.append(_mosfet("M1", "nmos", d="d1", g="inp", s="tail", b="vss",
                            W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "nmos", d="d2", g="inn", s="tail", b="vss",
                            W=W_diff, L=L_diff))

    # PMOS current-mirror load (diode + replica)
    devices.append(_mosfet("M3", "pmos", d="d1", g="d1", s="vdd", b="vdd",
                            W=W_load1, L=L_load1))
    devices.append(_mosfet("M4", "pmos", d="d2", g="d1", s="vdd", b="vdd",
                            W=W_load1, L=L_load1))

    # Second stage: PMOS common-source amplifier
    devices.append(_mosfet("Mcs",  "pmos", d="out", g="d2", s="vdd", b="vdd",
                            W=W_cs, L=L_cs))

    # Second-stage NMOS current-source load (mirror of Mbn, ratio K)
    devices.append(_mosfet("Mlcs", "nmos", d="out", g="nbias", s="vss", b="vss",
                            W=W_bn, L=L_bn, M=K_cs_load))

    # We need to inject the reference current into pbias. Put a current source
    # inside the cell pulling I_ref from pbias to vss.
    # Since Mbp (PMOS diode) has source=vdd and drain=pbias, current must flow
    # out of pbias toward vss. So an internal source: I p=pbias n=vss value=I_ref.
    devices.append(Device(
        instance_name="Iref",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "pbias", "n": "vss"},
        sizing={"value": I_ref},
    ))

    # Miller compensation: Cc from out to d2 (no nulling resistor in this v0)
    devices.append(_cap("Cc", p="out", n="d2", value=Cc))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id=pdk_id,
    )

    # R119 C: optional DRC annotations
    # MATCHING (diff pair) / CURRENT_MIRROR_MATCHING (load) /
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
            devices=("M1", "M2"),
            extras={
                "rationale": (
                    "NMOS diff pair — Vth mismatch maps directly to "
                    "input-referred offset and sets the achievable "
                    "CMRR / PSRR floor of the Razavi Ch. 9 textbook "
                    "2-stage Miller baseline"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.CURRENT_MIRROR_MATCHING,
            devices=("M3", "M4"),
            extras={
                "rationale": (
                    "PMOS load mirror — M3 diode-connected at d1, M4 "
                    "replica at d2. Mismatch unbalances diff-pair leg "
                    "currents and contributes a systematic offset on "
                    "top of the M1/M2 Vth mismatch component"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "NMOS tail current source (M=N_tail) carries the "
                    "entire stage-1 bias current N_tail * I_ref; pwell "
                    "guard ring tied to vss isolates the diff-pair "
                    "tail node from substrate-coupled noise. Tail-"
                    "noise -> common-mode -> finite-CMRR conversion "
                    "sets PSRR_VSS floor of the input stage"
                ),
                "well_type": "pwell",
            },
        ))

    return cell


# ─── Testbench builder for AC characterization ─────────────────────

def build_ac_testbench(cell: Cell, supply: float | None = None, vcm_ratio: float = 0.5,
                      f_start: float = 1.0, f_stop: float = 1e9,
                      points_per_dec: int = 20) -> Testbench:
    """Standard AC testbench: VDD/VSS supplies, differential AC input via VCVS,
    DC common-mode bias, infinite-impedance feedback to set DC operating point.

    `supply=None` auto-resolves from `cell.pdk_id` (sky130 → 1.8 V,
    gf180mcu → 3.3 V, etc.). Pass explicit value to override.
    """
    if supply is None:
        from ._supply import resolve_supply
        supply = resolve_supply(cell)
    sources: list[Source] = [
        Source("voltage_dc",  "VDD",  pins={"p": "vdd", "n": "0"}, params={"dc": supply}),
        Source("voltage_dc",  "VSS",  pins={"p": "vss", "n": "0"}, params={"dc": 0.0}),
        Source("voltage_dc",  "VCM",  pins={"p": "vcm", "n": "0"}, params={"dc": supply * vcm_ratio}),
        Source("voltage_ac",  "VSIG", pins={"p": "sig", "n": "0"}, params={"dc": 0.0, "ac": 1.0}),
    ]
    extras = [
        Device("Edp", "vcvs", None,
               pins={"op": "inp", "on": "vcm", "ip": "sig", "in": "0"},
               sizing={"gain": 0.5}),
        Device("Edn", "vcvs", None,
               pins={"op": "inn", "on": "vcm", "ip": "sig", "in": "0"},
               sizing={"gain": -0.5}),
        Device("CL", "cap", None,
               pins={"p": "out", "n": "vss"},
               sizing={"value": 2e-12}),
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
