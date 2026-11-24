"""
Built-in template: telescopic_cascode.

Single-stage telescopic cascode opamp (Razavi Chapter 9) for sky130 1.8V.

Topology — full stack from vdd to vss along each output branch:

           vdd ─────┬────────┬────────┬──────────────────────
                    │        │        │
                  Mload1_p Mload2_p Mload_diag    (PMOS current mirror,
                    │        │        │           gate=pbias_load, s=vdd)
                  d1a      outa     pbl_d  ── pbias_load (diode tap)
                    │        │        │
                  Ml1_pc   Ml2_pc   Mpc_diag      (PMOS cascode,
                    │        │        │           gate=pbias_cas, s=above)
                    d1       out     pbc_d  ── pbias_cas (diode tap)
                    │        │
                  Mcs1_n   Mcs2_n                 (NMOS cascode,
                    │        │                    gate=nbias_cas, s=below)
                    e1       e2
                    │        │
                    M1       M2                   (NMOS input diff pair,
                    │        │                    gate=inp/inn, s=tail)
                    └────┬───┘
                       tail
                         │
                       Mtail                      (NMOS tail current source,
                         │                        gate=nbias_bot, s=vss)
                        vss

Bias generation:
  A single reference current Iref flows top-down through a chain of four
  diode-connected MOSFETs from vdd to vss, tapping out four bias voltages:

      vdd → [Mload_diag: pmos diode]      → pbias_load
            [Mpc_diag:   pmos diode]      → pbias_cas
            [Mnc_diag:   nmos diode]      → nbias_cas
            [Mtail_diag: nmos diode]      → nbias_bot
            → vss

  The chain auto-balances: each diode sets one Vgs of headroom.

  Iref is injected by an internal cur_src I_p=vdd, n=nbias_bot — but we route
  the bias-chain current via a separate path so that Mtail_diag sees Iref. We
  use a single current source pushing Iref into the top of the chain (a PMOS
  current source from vdd or equivalently a current pull from pbias_load to
  vss). The simplest robust form: drive Iref directly into the top of the
  stack so that the same current flows through all four diodes in series.

  Implementation choice: the four diodes are wired source-to-drain in a
  single string from vdd to vss; Iref enters at the vdd end (the pmos
  source-to-drain via a current source) so the four diodes carry exactly
  Iref. This auto-generates the four taps with correct headroom on the
  1.8V sky130 supply.

Topology characteristics:
- Gain: high (single-stage cascode product (gm·ro)^2 ~ 70 dB on sky130 1.8V).
- Output swing: narrow (4 Vov, ~0.8 V on 1.8V supply).
- Compensation: single-stage, dominant pole at output node (CL || rout); no
  Miller capacitor needed.
- VCM range: roughly mid-supply (must keep both input pair and tail in sat).

Port order (must match build_ac_testbench): inn, inp, out, vdd, vss.
"""

from __future__ import annotations

import os

from ..ir import (
    AcAnalysis, Cell, Device, Measure, OpAnalysis, Source, Spec, Testbench,
)
from .two_stage_miller_pmos import build_ac_testbench   # re-exported


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
    """Resolve the nominal supply for sizing-time decisions.

    Same resolution order as `folded_cascode._resolve_vdd_nom`:
    explicit arg → `CLAY_PDK_ID` env-var lookup → 1.8 V fallback.
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


_HEADROOM_THRESHOLD_V = 2.0


def build_telescopic_cascode(
    cell_name: str,
    spec: Spec,
    *,
    # Reference current (sets all branch currents via mirroring ratios)
    I_ref: float = 20e-6,
    # Input differential pair (NMOS)
    W_diff: float = 40e-6,    L_diff: float = 0.5e-6,
    # NMOS cascode (above diff pair)
    W_nc:   float = 40e-6,    L_nc:   float = 0.5e-6,
    # PMOS cascode (below current-mirror top)
    W_pc:   float = 40e-6,    L_pc:   float = 0.5e-6,
    # PMOS current-mirror top (mirrors of pbias_load diode)
    W_load: float = 40e-6,    L_load: float = 1e-6,
    # Tail NMOS current source (mirror of nbias_bot diode)
    W_tail: float = 20e-6,    L_tail: float = 1e-6,
    # Bias-chain device sizes (these all carry exactly Iref; they are the
    # "unit" devices that the larger branch devices mirror).
    W_bload: float = 10e-6,   L_bload: float = 1e-6,    # PMOS diode at top
    W_bpc:   float = 10e-6,   L_bpc:   float = 0.5e-6,  # PMOS cascode diode
    W_bnc:   float = 10e-6,   L_bnc:   float = 0.5e-6,  # NMOS cascode diode
    W_btail: float = 5e-6,    L_btail: float = 1e-6,    # NMOS tail diode
    # Branch mirror ratio (how much branch current relative to Iref)
    M_branch: int = 4,        # each output branch carries M_branch * Iref / 2
                              # (factor of 2 because tail splits in two)
    M_tail:   int = 8,        # Mtail width multiplier (carries M_branch * Iref total)

    # ─── Intent (gm/Id) form for input diff pair ─────────────
    # When set, overrides W_diff via the closed-form solver in clay.sizing.
    # I_d through each NMOS diff-pair half = (M_branch * I_ref) / 2.
    gm_id_diff: float | None = None,

    # Nominal supply voltage; selects bias-sizing regime. When None,
    # resolved via `CLAY_PDK_ID` env var → PDK adapter vdd_nom →
    # fallback 1.8 V (sky130 default). On 3.3 V supplies the bias
    # devices get auto-resized (see body for details).
    vdd_nom: float | None = None,
    # R110 C: qualitative DRC annotation surface (additive metadata,
    # see folded_cascode.py R108 C precedent). Default False keeps
    # every pre-R110 caller byte-for-byte identical. When True, three
    # qualitative hints attach:
    #   MATCHING       on NMOS diff pair (M1 / M2),
    #   GUARD_RING     on NMOS tail current source (Mtail),
    #   WELL_ISOLATED  on the NMOS cascode pair (Mcs1_n / Mcs2_n) —
    #                  the telescopic-cascode topology signature.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for a single-stage telescopic cascode opamp.

    Args:
        cell_name: name for the .subckt.
        spec: target spec (carried into Cell.spec; used by verifier).
        I_ref: reference current through the bias chain.
        W_diff, L_diff: input diff pair sizing.
        W_nc, L_nc:     NMOS cascode sizing.
        W_pc, L_pc:     PMOS cascode sizing.
        W_load, L_load: PMOS load (mirror) sizing.
        W_tail, L_tail: tail current source sizing.
        W_b*, L_b*:     bias-chain diode unit-device sizing.
        M_branch:       PMOS mirror ratio for load devices (relative to Mload_diag).
        M_tail:         NMOS mirror ratio for tail device (relative to Mtail_diag).
        vdd_nom:        nominal supply (V). None → CLAY_PDK_ID env var
                        → 1.8 V fallback. Selects sky130 vs. high-supply
                        bias sizing defaults.
    """
    vdd_nom_v = _resolve_vdd_nom(vdd_nom)
    high_supply = vdd_nom_v > _HEADROOM_THRESHOLD_V

    # ─── Supply-adaptive bias-chain sizing ─────────────────────────────
    # The sky130 1v8 defaults were tuned to keep the cell in saturation
    # on a thin 1.8 V rail with the diff pair near subthreshold. On
    # gf180mcu 3.3 V the same sizes leave the PMOS load mirror with
    # Vov≈0.03 V (basically subthreshold), and ngspice's subckt `m=`
    # multiplier does not propagate through the foundry's pmos_3p3 /
    # sky130_fd_pr__pfet_01v8 wrapper, so M-multiplying the mirror
    # device doesn't recover the lost current. The workaround at high
    # supply: scale I_ref up ~5× and shrink the PMOS load+tail mirror
    # devices to ~5 μm. That moves the load mirror into strong inversion
    # at ~Vov=0.3 V, where the cascode stack has enough current
    # headroom to pull the output node toward mid-rail.
    if high_supply:
        if I_ref == 20e-6:        # untouched default → auto-scale
            I_ref = 100e-6
        if W_load == 40e-6:
            W_load = 5e-6
        if W_tail == 20e-6:
            W_tail = 5e-6
        # v0.0.49 PVT re-tune: at gf180mcu 3v3 the v0.0.48 W_diff=40u
        # gives the NMOS diff pair a low |Vov| (~0.15 V). At fast-NMOS
        # corners (ff, fs) Vth drops faster than μ_n's effect on the
        # diff pair → V_tail rises → e1/e2 rise → NMOS cascode Vds
        # squeezed → cascode in triode → output collapses. The fix
        # narrows the diff pair to W=10u so |Vov| widens to ~0.3 V,
        # pulling V_tail lower and giving the NMOS cascode the
        # Vds_sat budget it needs across all 5 corners. The diff pair
        # also moves further from subthreshold which improves AC gain
        # on the working corners.
        if W_diff == 40e-6:
            W_diff = 10e-6

    # gm/Id intent override: derive W_diff from (gm_id_diff, I_d, L_diff).
    # I_d through each NMOS diff-pair half = (M_branch * I_ref) / 2.
    if gm_id_diff is not None:
        from ..sizing import closed_form_W
        I_d_diff = (M_branch * I_ref) / 2.0
        W_diff = closed_form_W("nmos", gm_id=gm_id_diff,
                               I_d=I_d_diff, L=L_diff).W

    ports = ["inn", "inp", "out", "vdd", "vss"]
    devices: list[Device] = []

    # ─── Bias chain ─────────────────────────────────────────────
    # Four diode-connected MOSFETs stacked from vdd to vss, all carrying Iref
    # so each one's Vgs is well-defined and consistent for the mirror devices.
    #
    # Stack (top → bottom):
    #   vdd → Mload_diag (pmos diode, s=vdd, d=pbias_load, g=pbias_load)
    #       → Mpc_diag   (pmos diode, s=pbias_load, d=pbias_cas, g=pbias_cas)
    #       → Mnc_diag   (nmos diode, d=nbias_cas, s=mid_n, g=nbias_cas)
    #       → Mtail_diag (nmos diode, d=mid_n, s=vss, g=nbias_bot)
    #
    # But to keep mirrors aligned we need pbias_cas = pbias_load - |Vgs_p|,
    # so the PMOS-cascode diode source = pbias_load (its drain at vdd side),
    # and drain = pbias_cas (one Vgs lower toward vss).
    #
    # On the NMOS side: Mnc_diag drain = nbias_cas (the cascode-gate tap),
    # source = nbias_bot (which is also the tail diode's drain), gate=nbias_cas.
    # Mtail_diag drain=nbias_bot, source=vss, gate=nbias_bot.
    #
    # The two stacks must be connected: the PMOS-stack bottom (pbias_cas) and
    # NMOS-stack top (nbias_cas) cannot be the same node, because then there
    # would be no headroom budget. Instead we drive each side with its own
    # current source.
    #
    # Cleanest approach: TWO independent bias legs, each carrying Iref:
    #   leg A (PMOS):  vdd → Mload_diag (diode) → Mpc_diag (diode) → Iref→vss
    #   leg B (NMOS):  Iref→vdd → Mnc_diag (diode) → Mtail_diag (diode) → vss
    #
    # leg A generates pbias_load and pbias_cas.
    # leg B generates nbias_cas and nbias_bot.

    # ── leg A: PMOS bias diode stack ──
    # vdd → Mload_diag → pbias_load → Mpc_diag → pbias_cas → (Iref sink to vss)
    devices.append(_mosfet(
        "Mload_diag", "pmos",
        d="pbias_load", g="pbias_load", s="vdd", b="vdd",
        W=W_bload, L=L_bload,
    ))
    # Body tied to its own source (separate n-well) so the |Vgs_p| drop
    # of this stacked diode does NOT include the body-effect shift. With
    # body=vdd, Vsb = vdd - pbias_load > 0 raises |Vth_p| by ~0.3 V on
    # 3.3 V supplies, pulling pbias_cas down into a region where the
    # PMOS cascode mirrors run in deep triode and the output node pins
    # at vdd. Tying body=source removes that shift.
    devices.append(_mosfet(
        "Mpc_diag", "pmos",
        d="pbias_cas", g="pbias_cas", s="pbias_load", b="pbias_load",
        W=W_bpc, L=L_bpc,
    ))
    # Iref pulls current from pbias_cas to vss (so it flows top-down through both diodes).
    # cur_src convention: current flows internally from p → n; pulling current
    # *out* of pbias_cas to vss means setting p=pbias_cas, n=vss.
    devices.append(Device(
        instance_name="Iref_A",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "pbias_cas", "n": "vss"},
        sizing={"value": I_ref},
    ))

    # ── leg B: NMOS bias diode stack ──
    # vdd → (Iref source) → nbias_cas → Mnc_diag → nbias_bot → Mtail_diag → vss
    devices.append(_mosfet(
        "Mnc_diag", "nmos",
        d="nbias_cas", g="nbias_cas", s="nbias_bot", b="vss",
        W=W_bnc, L=L_bnc,
    ))
    devices.append(_mosfet(
        "Mtail_diag", "nmos",
        d="nbias_bot", g="nbias_bot", s="vss", b="vss",
        W=W_btail, L=L_btail,
    ))
    # Iref pushes current into nbias_cas from vdd.
    # Internal cur_src direction: p → n, so p=vdd, n=nbias_cas means current
    # enters nbias_cas from vdd; this is exactly the bias drive.
    devices.append(Device(
        instance_name="Iref_B",
        device_class="cur_src",
        pdk_variant=None,
        pins={"p": "vdd", "n": "nbias_cas"},
        sizing={"value": I_ref},
    ))

    # ─── Tail current source ────────────────────────────────────
    # NMOS mirror of Mtail_diag, ratio M_tail. Total branch current = M_tail * Iref.
    devices.append(_mosfet(
        "Mtail", "nmos",
        d="tail", g="nbias_bot", s="vss", b="vss",
        W=W_tail, L=L_tail, M=M_tail,
    ))

    # ─── Input differential pair (NMOS) ─────────────────────────
    # Sources tied at tail node; drains feed the cascode below.
    devices.append(_mosfet(
        "M1", "nmos",
        d="e1", g="inp", s="tail", b="vss",
        W=W_diff, L=L_diff,
    ))
    devices.append(_mosfet(
        "M2", "nmos",
        d="e2", g="inn", s="tail", b="vss",
        W=W_diff, L=L_diff,
    ))

    # ─── NMOS cascode (above diff-pair drains) ──────────────────
    # Common-gate; gate=nbias_cas; sources = M1/M2 drains; drains = d1/out.
    devices.append(_mosfet(
        "Mcs1_n", "nmos",
        d="d1", g="nbias_cas", s="e1", b="vss",
        W=W_nc, L=L_nc,
    ))
    devices.append(_mosfet(
        "Mcs2_n", "nmos",
        d="out", g="nbias_cas", s="e2", b="vss",
        W=W_nc, L=L_nc,
    ))

    # ─── PMOS cascode (below current-mirror top) ────────────────
    # Common-gate; gate=pbias_cas; sources = mirror-top drains; drains = d1/out.
    devices.append(_mosfet(
        "Ml1_pc", "pmos",
        d="d1", g="pbias_cas", s="d1a", b="vdd",
        W=W_pc, L=L_pc,
    ))
    devices.append(_mosfet(
        "Ml2_pc", "pmos",
        d="out", g="pbias_cas", s="outa", b="vdd",
        W=W_pc, L=L_pc,
    ))

    # ─── PMOS current-mirror top ────────────────────────────────
    # Two mirrors of Mload_diag, gates tied to pbias_load. Each carries
    # M_branch * Iref. Sources at vdd; drains feed PMOS cascode sources.
    devices.append(_mosfet(
        "Mload1_p", "pmos",
        d="d1a", g="pbias_load", s="vdd", b="vdd",
        W=W_load, L=L_load, M=M_branch,
    ))
    devices.append(_mosfet(
        "Mload2_p", "pmos",
        d="outa", g="pbias_load", s="vdd", b="vdd",
        W=W_load, L=L_load, M=M_branch,
    ))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id="sky130",
    )

    # ─── R110 C: qualitative DRC annotation surface ────────────────────
    # R108 C established the surface on folded_cascode; R109 C extended
    # to 3 NMCF-family siblings. R110 C broadens to a single-stage
    # telescopic cascode — Razavi Ch. 9, NMOS-input variant. Annotations
    # live only in cell.__dict__["_drc_annotations"]; SPICE emit (which
    # iterates cell.devices) is byte-for-byte unaffected when default
    # attach_drc=False.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        # 1) MATCHING — NMOS differential pair (M1 / M2). The telescopic
        #    cascode has an NMOS input pair (unlike folded_cascode's
        #    PMOS pair); input offset is dominated by M1/M2 Vth-mismatch.
        #    Common-centroid / interdigitated layout per Hastings Ch. 8.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("M1", "M2"),
            extras={
                "rationale": "NMOS diff pair — input offset is Vth-mismatch limited",
                "ratio": "1:1",
            },
        ))

        # 2) GUARD_RING — NMOS tail current source (Mtail). The tail
        #    carries M_tail × I_ref and sits at the bottom of the diff
        #    pair; substrate noise injected via the p-bulk converts to
        #    a common-mode disturbance on the tail node and degrades
        #    CMRR. A dedicated pwell guard ring keeps neighboring
        #    switching activity from modulating the tail current.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": "tail current source — substrate-noise isolation",
                "well_type": "pwell",
            },
        ))

        # 3) WELL_ISOLATED — NMOS cascode pair (Mcs1_n / Mcs2_n). The
        #    telescopic-cascode signature: NMOS cascode devices above
        #    the diff pair (gates tied to nbias_cas, sources to M1/M2
        #    drains). Their drain-to-bulk capacitance couples output
        #    swing onto the substrate at the high-Z output node (out),
        #    and body-effect from a shared pwell modulates |Vth_n|
        #    asymmetrically when the differential output swings. A
        #    dedicated pwell tie (or deep-nwell isolation) for the
        #    cascode pair restores the (gm·ro)^2 output impedance the
        #    telescopic topology relies on.
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.WELL_ISOLATED,
            devices=("Mcs1_n", "Mcs2_n"),
            extras={
                "rationale": "NMOS telescopic cascode pair — body-effect isolation at high-Z output node",
                "isolate_from": ("M1", "M2"),
            },
        ))

    return cell


__all__ = ["build_telescopic_cascode", "build_ac_testbench"]
