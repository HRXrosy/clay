"""
Built-in template: folded_cascode.

Single-stage folded-cascode opamp, PMOS-input, multi-PDK (textbook Razavi
Chapter 9, fully-cascoded single-output variant).

Topology
========

                   vdd
        ┌──────┬─────────┬─────────┐
        │      │         │         │
      Mtail Mload1_p   Mload2_p   (PMOS, gate=pbias_load)
        │      │         │
       tail   a1        a2        (top of PMOS cascode)
        │      │         │
   M1 ──┤      │         │
   M2 ──┤    Mload1c   Mload2c    (PMOS cascode, gate=pbias_cas)
        │      │         │
        ▼      ▼         ▼
              d1=fp     d2=out    (folded summing nodes)
                │         │
              Mcs1_n    Mcs2_n    (NMOS cascode, gate=nbias_cas)
                │         │
                b1        b2
                │         │
              Mbot1     Mbot2     (NMOS, gate=nbias_bot)
                │         │
               vss       vss

Diff pair
---------
M1: source=tail, gate=inp, drain=fp    (PMOS; unused fold side)
M2: source=tail, gate=inn, drain=out   (PMOS; output fold side)

Bias plan — supply-adaptive
---------------------------
The folded cascode needs four bias rails: pbias_load (PMOS mirror gate),
pbias_cas (PMOS cascode gate), nbias_cas (NMOS cascode gate), nbias_bot
(NMOS bottom mirror gate). The exact bias generation depends on what
supply the cell will be operated at, which is set by the PDK adapter
(`vdd_nom`).

Two regimes:

  (A) Low-supply (vdd_nom <= 2.0 V, e.g. sky130 1v8)
      sky130's |Vthp| ≈ 1.0 V at L=1u so two stacked PMOS diodes (~2.0 V)
      do NOT fit on a 1.8 V rail. We fall back to single-diode bias for
      pbias_load and rail-tie the cascode gates (pbias_cas=vss,
      nbias_cas=vdd). The cascode is heavily ON; KCL at the fold node
      sets the cascode Vds in the saturated region under matched
      currents. Achievable DC gain is modest (~10 dB on sky130) but the
      cell builds correctly and the diff pair drives a high-impedance
      summing node.

  (B) Headroom-OK (vdd_nom > 2.0 V, e.g. gf180mcu 3v3)
      Two-stage diode bias chains fit comfortably:
          PMOS: vdd → MbpT(diode) → pbias_load → MbpC(diode) →
                pbias_cas → Iref_p → vss
          NMOS: vdd → Iref_n → nbias_cas → MbnC(diode) →
                nbias_bot → MbnB(diode) → vss
      pbias_load sits one |Vgs_p| below vdd; pbias_cas one |Vgs_p|
      below pbias_load. nbias_bot sits one Vgs_n above vss; nbias_cas
      one Vgs_n above nbias_bot. With these bias values, the PMOS
      cascode (gate=pbias_cas, source=a1) and NMOS cascode (gate=
      nbias_cas, source=b1) both sit in saturation and the load /
      bottom mirror devices retain their cascode-improved output
      impedance — yielding the textbook (gm·ro)^2 single-stage gain.

Supply regime selection
-----------------------
The builder accepts `vdd_nom` as a kwarg. When None (default), it tries
to discover the supply by reading `CLAY_PDK_ID` from the environment
and looking up that PDK adapter's `vdd_nom`. If the env var is unset or
the lookup fails, it falls back to 1.8 V — matching the sky130 default
and preserving every pre-v0.0.42 sky130 result.

Callers can also pass `vdd_nom=3.3` directly for stress-testing a
sky130 cell at the higher supply, or to force the headroom-OK chain.
"""

from __future__ import annotations

import os

from ..ir import Cell, Device, Spec
# Re-export the existing AC testbench builder so callers can use the same
# differential-VCVS rig that ships with two_stage_miller_pmos.
from .two_stage_miller_pmos import build_ac_testbench  # noqa: F401


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


def _resolve_vdd_nom(explicit: float | None) -> float:
    """Return the nominal supply voltage to size the bias chain for.

    Resolution order:
      1. `explicit` argument (caller override).
      2. `CLAY_PDK_ID` env var → look up `clay.pdk.get_pdk(...).vdd_nom`.
      3. Fall back to 1.8 V (sky130 default).
    """
    if explicit is not None:
        return float(explicit)
    pdk_env = os.environ.get("CLAY_PDK_ID")
    if pdk_env:
        try:
            from ..pdk import get_pdk
            return float(get_pdk(pdk_env).vdd_nom)
        except Exception:
            pass
    return 1.8


# Headroom threshold (V) above which the 2-diode-stack bias chain fits.
# 2 * (|Vth_p| + Vov) ≈ 1.6-1.8 V on common foundry processes; setting
# the trip at 2.0 V leaves a small margin so 1.8 V-class supplies fall
# safely into the low-supply branch while 3.3 V supplies pick the
# high-supply branch.
_HEADROOM_THRESHOLD_V = 2.0


def build_folded_cascode(
    cell_name: str,
    spec: Spec,
    *,
    # Bias reference current
    I_ref: float = 20e-6,
    # Bias-chain device sizing (each leg carries I_ref). These must be wide
    # enough that each diode-connected device passes I_ref at moderate Vov
    # (~0.15 V). sky130 PFET has |Vthp|≈1.03 V and lower mobility than NFET
    # so the PMOS bias device needs to be ~4× wider than the NMOS one to
    # mirror the same current at the same Vov. The current-source mirrors
    # below (Mtail, Mload, Mbot) re-use these same widths so that
    # M-multiplier fully controls the current ratio (mirror ratio = M).
    W_bp: float = 80e-6,
    L_bp: float = 1e-6,
    W_bn: float = 20e-6,
    L_bn: float = 1e-6,
    # Cascode-diode device sizes for the headroom-OK (vdd >= 2 V) regime.
    # The cascode diode sees a tail current of I_ref but stacks ON TOP of
    # the mirror diode (PMOS side) / BELOW the mirror diode (NMOS side),
    # so its absolute Vgs determines `pbias_cas` / `nbias_cas` headroom.
    # Sized at the same W as the mirror diodes by default so the chain
    # generates roughly two equal Vgs drops.
    W_bp_cas: float = 80e-6,
    L_bp_cas: float = 1e-6,
    W_bn_cas: float = 20e-6,
    L_bn_cas: float = 1e-6,
    # Mirror ratios off the bias diodes (M-multiplier on the cell's transistors).
    #
    # On sky130 1v8 with |Vthp|≈1.03V, a PMOS diff pair at VCM=0.9V cannot
    # saturate (it would need Vtail > 1.93V > vdd). The pair therefore runs
    # in weak inversion / subthreshold. To keep the fold node out of a rail
    # under that condition, we set N_load = N_bot so the PMOS load current
    # source and NMOS bottom current source cancel at DC; the diff pair
    # then injects only the small subthreshold residual onto a high-impedance
    # cascoded summing node, giving a usable AC gain even at tiny DC current.
    N_tail: int = 2,           # I_tail = N_tail * I_ref (target in saturation)
    N_load: int = 2,           # I_load = N_load * I_ref per branch
    N_bot: int = 2,            # I_bot  = N_bot  * I_ref per branch
                               # Balance: I_bot ≈ I_load (diff pair runs sub-Vth)
    # Diff pair (PMOS) — explicit W form
    W_diff: float = 20e-6,
    L_diff: float = 0.5e-6,
    # Diff pair sizing intent (gm/Id form) — overrides W_diff when set.
    gm_id_diff: float | None = None,
    # PMOS cascode device (sized for the cascode current, not a mirror)
    W_load_p_cas: float = 80e-6,
    L_load_p_cas: float = 0.5e-6,
    # NMOS cascode (also sized for cascode current, not a mirror)
    W_cs_n: float = 20e-6,
    L_cs_n: float = 0.5e-6,
    # Nominal supply voltage; selects bias-chain regime. When None,
    # resolved via `CLAY_PDK_ID` env var → PDK adapter vdd_nom →
    # fallback 1.8 V (sky130 default).
    vdd_nom: float | None = None,
    # R108 C: qualitative DRC annotation surface (additive metadata
    # attached via clay.pdks._drc_annotations). Defaults to False so
    # every pre-R108 invocation produces a byte-identical Cell (no
    # `_drc_annotations` key in __dict__, no behavior change in SPICE
    # emit / sizing / verifier). When True, three qualitative hints are
    # attached: MATCHING on the PMOS diff pair (M1/M2), GUARD_RING on the
    # PMOS tail current source (Mtail), and WELL_ISOLATED on the PMOS
    # cascode mirror pair (Mload1_p_cas/Mload2_p_cas). Layout prompt
    # consumes these via clay.pdks.export_drc_manifest.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a single-stage folded-cascode opamp.

    Port order is the standard differential-amp convention used by the AC
    testbench builder: ["inn", "inp", "out", "vdd", "vss"].
    """
    vdd_nom_v = _resolve_vdd_nom(vdd_nom)
    high_supply = vdd_nom_v > _HEADROOM_THRESHOLD_V

    # ─── Supply-adaptive bias-chain sizing ─────────────────────────────
    # The sky130 1v8 defaults (W_bp=80u, W_bn=20u, I_ref=20uA) keep the
    # diff pair in weak inversion (necessary because |Vth_p|≈1V leaves
    # almost no headroom on a 1.8 V rail). On gf180mcu 3.3 V those same
    # numbers leave the PMOS load mirror's Vov at ~0.03 V — barely above
    # the subthreshold knee. ngspice's subckt `m=` multiplier doesn't
    # propagate to the inner `m0` device in the foundry's PDK wrapper
    # (gf180mcu's `pmos_3p3` / sky130's `sky130_fd_pr__pfet_01v8` both
    # ignore `m` at the subckt boundary), so we can't recover the
    # current via M-multiplication on the cell-level transistors. The
    # workaround at high supply: scale I_ref up ~25× and shrink the
    # bias diodes ~16× so each diode runs at moderate Vov (~0.4 V).
    # This brings the load/bot mirrors out of subthreshold and into
    # the strong-inversion regime where the cascode stack has enough
    # current to pull the output node off the rail.
    #
    # Callers can still pass `I_ref=`, `W_bp=`, `W_bn=` explicitly to
    # override. Defaults react to vdd_nom so a plain Clay file with
    # only `pdk 'gf180mcu';` gets reasonable sizing automatically.
    if high_supply:
        if I_ref == 20e-6:        # untouched default → auto-scale
            I_ref = 500e-6
        if W_bp == 80e-6:
            W_bp = 5e-6
        if W_bn == 20e-6:
            W_bn = 5e-6
        # v0.0.49 PVT re-tune: v0.0.48 kept W_bp_cas / W_bn_cas at 5u
        # so |Vgs| of the second-stage diode matched the first. At
        # 125 °C the NMOS sink overshoots the PMOS source on slow-PMOS
        # corners (tt/ss/sf/fs at 125 °C death-biased). Shrinking
        # the cascode-bias diodes pushes pbias_cas further from vdd
        # and nbias_cas further from vss, widening the cascode-device
        # Vds_sat margin so each PMOS / NMOS cascode keeps its
        # output impedance at hot+slow.
        if W_bp_cas == 80e-6:
            W_bp_cas = 2e-6
        if W_bn_cas == 20e-6:
            W_bn_cas = 2e-6
        # v0.0.49 PVT re-tune: shorten the cascode-device L from
        # 0.5 μm → 0.35 μm so each cascode has more gm and a smaller
        # Vds_sat budget. The cascode output impedance trade-off
        # (shorter L → lower r_o) is small at 0.35 μm on gf180mcu
        # but the headroom-recovery is decisive at 125 °C.
        if L_load_p_cas == 0.5e-6:
            L_load_p_cas = 0.35e-6
        if L_cs_n == 0.5e-6:
            L_cs_n = 0.35e-6
        # v0.0.49: widen the PMOS diff pair so it operates at lower
        # |Vov| (lower |Vgs|), pulling the tail node further from vdd
        # and giving Mtail more saturation budget on hot/slow PMOS.
        if W_diff == 20e-6:
            W_diff = 80e-6

    # gm/Id intent override: derive W_diff from (gm_id_diff, I_d, L_diff).
    # I_d through each PMOS diff-pair half = I_tail / 2 = (N_tail * I_ref) / 2.
    if gm_id_diff is not None:
        from ..sizing import closed_form_W
        I_d_diff = (N_tail * I_ref) / 2.0
        W_diff = closed_form_W("pmos", gm_id=gm_id_diff,
                               I_d=I_d_diff, L=L_diff).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Bias leg (a): PMOS chain (single or 2-diode stack) ───────────
    # Low-supply: vdd → MbpT (diode) → pbias_load → [Iref_pl] → vss
    #   pbias_cas is wired directly to vss when instancing cascode devices.
    # High-supply: vdd → MbpT (diode) → pbias_load → MbpC (diode)
    #                → pbias_cas → [Iref_pl] → vss
    devices.append(_mosfet("MbpT", "pmos",
                           d="pbias_load", g="pbias_load", s="vdd", b="vdd",
                           W=W_bp, L=L_bp))
    if high_supply:
        # Second PMOS diode stacked below pbias_load; its drain/gate is
        # pbias_cas (one |Vgs_p| further down). Body tied to its own
        # source (separate n-well) so the |Vgs_p| drop does NOT include
        # the body-effect shift — keeping pbias_cas close to the level
        # the mirror cascode device needs to stay in saturation. (If
        # body were tied to vdd, Vsb = vdd - pbias_load > 0 would raise
        # |Vth_p| by γ·(sqrt(2φ + Vsb) - sqrt(2φ)), shifting pbias_cas
        # ~0.3 V lower than intended and dropping the cascode source
        # into deep triode.)
        devices.append(_mosfet("MbpC", "pmos",
                               d="pbias_cas", g="pbias_cas", s="pbias_load",
                               b="pbias_load",
                               W=W_bp_cas, L=L_bp_cas))
        # Sink pulls Iref out of the bottom of the chain.
        devices.append(Device(
            instance_name="Iref_pl",
            device_class="cur_src",
            pdk_variant=None,
            pins={"p": "pbias_cas", "n": "vss"},
            sizing={"value": I_ref},
        ))
    else:
        devices.append(Device(
            instance_name="Iref_pl",
            device_class="cur_src",
            pdk_variant=None,
            pins={"p": "pbias_load", "n": "vss"},
            sizing={"value": I_ref},
        ))

    # ─── Bias leg (b): NMOS chain (single or 2-diode stack) ───────────
    # Low-supply: vdd → [Iref_n] → nbias_bot → MbnB (diode) → vss
    #   nbias_cas is wired directly to vdd when instancing cascode devices.
    # High-supply: vdd → [Iref_n] → nbias_cas → MbnC (diode)
    #                → nbias_bot → MbnB (diode) → vss
    if high_supply:
        devices.append(Device(
            instance_name="Iref_n",
            device_class="cur_src",
            pdk_variant=None,
            pins={"p": "vdd", "n": "nbias_cas"},
            sizing={"value": I_ref},
        ))
        # Upper NMOS diode: drain/gate=nbias_cas, source=nbias_bot.
        devices.append(_mosfet("MbnC", "nmos",
                               d="nbias_cas", g="nbias_cas", s="nbias_bot",
                               b="vss",
                               W=W_bn_cas, L=L_bn_cas))
        # Lower NMOS diode: drain/gate=nbias_bot, source=vss.
        devices.append(_mosfet("MbnB", "nmos",
                               d="nbias_bot", g="nbias_bot", s="vss", b="vss",
                               W=W_bn, L=L_bn))
    else:
        devices.append(Device(
            instance_name="Iref_n",
            device_class="cur_src",
            pdk_variant=None,
            pins={"p": "vdd", "n": "nbias_bot"},
            sizing={"value": I_ref},
        ))
        devices.append(_mosfet("MbnB", "nmos",
                               d="nbias_bot", g="nbias_bot", s="vss", b="vss",
                               W=W_bn, L=L_bn))

    # ─── Cascode-gate net names (regime-dependent rail-tie) ───────────
    # In high-supply mode, pbias_cas and nbias_cas are dedicated bias
    # nets driven by the diode chains above. In low-supply mode, the
    # cascode gates rail-tie to the opposite rail of their device kind
    # (sky130 fallback documented above), so we directly wire vss/vdd.
    pbias_cas_net = "pbias_cas" if high_supply else "vss"
    nbias_cas_net = "nbias_cas" if high_supply else "vdd"

    # ─── Tail current source (PMOS) ───────────────────────────────────
    devices.append(_mosfet("Mtail", "pmos",
                           d="tail", g="pbias_load", s="vdd", b="vdd",
                           W=W_bp, L=L_bp, M=N_tail))

    # ─── PMOS differential pair ───────────────────────────────────────
    # M1 drain → fp (folded-positive, unused side)
    # M2 drain → out (folded-negative; this IS the output node)
    devices.append(_mosfet("M1", "pmos",
                           d="fp", g="inp", s="tail", b="vdd",
                           W=W_diff, L=L_diff))
    devices.append(_mosfet("M2", "pmos",
                           d="out", g="inn", s="tail", b="vdd",
                           W=W_diff, L=L_diff))

    # ─── PMOS cascoded load (top of folded nodes) ─────────────────────
    devices.append(_mosfet("Mload1_p", "pmos",
                           d="a1", g="pbias_load", s="vdd", b="vdd",
                           W=W_bp, L=L_bp, M=N_load))
    devices.append(_mosfet("Mload2_p", "pmos",
                           d="a2", g="pbias_load", s="vdd", b="vdd",
                           W=W_bp, L=L_bp, M=N_load))
    devices.append(_mosfet("Mload1_p_cas", "pmos",
                           d="fp", g=pbias_cas_net, s="a1", b="vdd",
                           W=W_load_p_cas, L=L_load_p_cas, M=N_load))
    devices.append(_mosfet("Mload2_p_cas", "pmos",
                           d="out", g=pbias_cas_net, s="a2", b="vdd",
                           W=W_load_p_cas, L=L_load_p_cas, M=N_load))

    # ─── NMOS cascode (bottom of folded nodes) ────────────────────────
    devices.append(_mosfet("Mcs1_n", "nmos",
                           d="fp", g=nbias_cas_net, s="b1", b="vss",
                           W=W_cs_n, L=L_cs_n, M=N_bot))
    devices.append(_mosfet("Mcs2_n", "nmos",
                           d="out", g=nbias_cas_net, s="b2", b="vss",
                           W=W_cs_n, L=L_cs_n, M=N_bot))

    # ─── NMOS bottom current sources (mirror MbnB) ────────────────────
    # Same W/L as MbnB so the M-multiplier fully controls mirror ratio.
    devices.append(_mosfet("Mbot1", "nmos",
                           d="b1", g="nbias_bot", s="vss", b="vss",
                           W=W_bn, L=L_bn, M=N_bot))
    devices.append(_mosfet("Mbot2", "nmos",
                           d="b2", g="nbias_bot", s="vss", b="vss",
                           W=W_bn, L=L_bn, M=N_bot))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R108 C: qualitative DRC annotation surface ────────────────────
    # When `attach_drc=True`, attach three template-side hints so the
    # downstream layout prompt can pick a matched / guarded / well-isolated
    # layout strategy. Annotations are stored in `cell.__dict__`, so the
    # SPICE emit path (which iterates `cell.devices` only) is byte-for-byte
    # unaffected. Default (attach_drc=False) means zero metadata attached —
    # every pre-R108 caller continues to receive an identical Cell.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — PMOS differential pair (M1 / M2). Layout prompt
        #    should pick a common-centroid or interdigitated geometry; the
        #    pair's Vth mismatch directly sets the amplifier's input
        #    offset, so this is the single most matching-sensitive group
        #    in the cell.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "PMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — PMOS tail current source (Mtail). The tail
        #    carries N_tail × I_ref and sits at the top of a high-Z tail
        #    node feeding both diff-pair sources; substrate noise coupled
        #    into Mtail propagates as common-mode disturbance into the
        #    summing nodes via the cascoded load mirror. A dedicated
        #    nwell guard ring keeps neighboring switching activity from
        #    modulating the tail current.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "nwell",
            },
        ))

        # 3) WELL_ISOLATED — PMOS cascode mirror pair (Mload1_p_cas /
        #    Mload2_p_cas). The cascode devices set the cell's output
        #    impedance (gm·ro)^2 in the headroom-OK regime; sharing a
        #    well with the load mirror diodes Mload1_p / Mload2_p exposes
        #    them to body-effect coupling that degrades ro at high
        #    output swing. A dedicated nwell tie (independent of the
        #    mirror diode's well) keeps the cascode body at vdd while
        #    preserving the mirror diode's separate body bias.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mload1_p_cas", "Mload2_p_cas"),
            extras={
                "rationale": "PMOS cascode pair — body-effect isolation from mirror diodes",
                "isolate_from": ("Mload1_p", "Mload2_p"),
            },
        ))

    return cell
