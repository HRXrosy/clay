"""
Built-in template: lc_tank_vco.

A cross-coupled NMOS LC-tank voltage-controlled oscillator. The first
Clay stage template that exercises the `ind` device primitive (v0.0.55)
end-to-end through a real circuit topology — passive LC tank + cross-
coupled gm pair + tail current source.

Topology (cross-coupled differential NMOS LC-VCO, textbook Razavi
Chapter 14 §1.2.2)
==================================================================

                       vdd
                        │
        ┌───────────────┼───────────────┐
        │               │               │
       L1             L2             (split inductor; the tank is
        │               │             actually a single center-tapped
        │               │             inductor whose virtual ground is
        │               │             vdd. In Clay we model it as two
        │               │             matched halves L_tank/2 between
        │               │             the differential outputs and vdd.)
        │               │
       outp ───────────── outn
        │      C_tank      │       (tank capacitor differential between
        │                  │        outp and outn)
        ├───┐          ┌───┤
        │   │          │   │
       Mosc1│  Mosc2   │   │
        │   ├──────────┘   │       (cross-coupled NMOS pair:
        │   │          ┌───┤        Mosc1.gate = outn, drain = outp
        │   │          │   │        Mosc2.gate = outp, drain = outn)
        └───┘          └───┘
              │
            tail (common source node)
              │
            Mtail (gate=vctrl, drain=tail, source=vss)
              │
             vss

Pin map of the cross-coupled NMOS pair:
  * Mosc1:  drain = outp, gate = outn, source = tail, body = vss
  * Mosc2:  drain = outn, gate = outp, source = tail, body = vss

Each MOSFET's drain is the opposite output, the gate is the same output.
That's the cross-coupled connection: looking into the drain of Mosc1
from outp, the small-signal resistance is −1/gm_osc (negative). Two
such devices in parallel see a combined −2/gm_osc impedance across the
tank — the "negative resistance" that compensates the tank's finite Q
and sustains oscillation.

Frequency:  f_osc = 1 / (2π √(L_tank · C_tank))

Each half of the differential tank carries inductance L_tank/2; when
the tank is driven differentially the two halves combine in series for
the differential mode, so the effective tank inductance for f_osc is
L_tank (full value).  We expose `L_tank` as a single design knob and
split it into two L_tank/2 elements internally to keep the topology
truly differential.

Why a single port-bound output is preserved
-------------------------------------------
The PLL plumbing in v0.0.40+ expects a single-ended `clk_out` port for
the ring VCO. To stay compatible with that wrapper, we expose
`clk_out = outp` as the single output port. The differential pair
internally oscillates between `outp` and `outn` 180° out of phase;
`clk_out` picks one phase.

Tuning
------
The `vctrl` port gates the tail NMOS (`Mtail`). Higher vctrl → more
tail current → faster oscillation startup AND slightly higher f_osc
(through the gm-dependence of the cross-coupled pair's input
capacitance). True frequency tuning needs a varactor in the tank, which
no PDK adapter currently models; for v0.0.56 we treat vctrl as an
amplitude / bias control. A follow-up varactor primitive would slot
into the tank as a voltage-dependent cap.

Sizing notes
------------
* `W_osc`, `L_osc`: width / length of the cross-coupled NMOS pair.
  Bigger W_osc gives more gm_osc and easier startup but more parasitic
  capacitance loading the tank, lowering Q. 20 µm / 0.5 µm is a balanced
  starting point on sky130 1v8 / gf180mcu 3v3.
* `I_bias`: nominal tail current. The startup condition is
  gm_osc · R_tank > 2 (per device, i.e. gm_osc > 2/R_tank). At
  I_bias = 1 mA and W = 20 µm, sky130 NMOS gives gm_osc ≈ 4 mS.
  With R_tank ≈ 200 Ω (Q ≈ 12 tank at 200 MHz with L=10 nH) that
  satisfies the criterion by ~4× — comfortable margin.
* `L_tank`, `C_tank`: tank reactive elements. For a 200 MHz target,
  L = 10 nH, C = 63 pF gives f ≈ 200 MHz (1/(2π·sqrt(10e-9·63e-12))).
  Real sky130 / gf180mcu inductors top out at ~10 nH on-die; off-chip
  test setups can use higher values to push f_osc lower (useful for
  ngspice convergence at MHz timescales, where the transient step is
  large enough to resolve cycles in a 5 µs window).

Honest scope
------------
This template's primary contract is "compile-success + correct device
list + correct SPICE emission". Achieving sustained oscillation in
ngspice without a noise / initial-condition kick is hard — the
cross-coupled pair sits in a metastable balanced state at DC OP and
ngspice batch mode rarely breaks symmetry on its own. v0.0.56 ships
the cell + structural tests; transient oscillation in a real testbench
is a v0.0.57+ item.

Port list (cross-coupled differential VCO)
------------------------------------------
['clk_out', 'vctrl', 'vdd', 'vss']

`clk_out` is `outp` (one phase of the differential output). The other
phase (`outn`) is an internal net.
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


def _ind(name: str, p: str, n: str, value: float) -> Device:
    return Device(
        instance_name=name,
        device_class="ind",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"value": value},
    )


def _varactor(name: str, p: str, n: str, W: float, L: float,
              M: int = 1) -> Device:
    return Device(
        instance_name=name,
        device_class="varactor",
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"W": W, "L": L, "M": float(M)},
    )


def _varactor_accum(name: str, p: str, n: str, W: float, L: float,
                    M: int = 1, flavor: str = "lvt") -> Device:
    """Build an accumulation-mode N-well varactor device.

    Distinct from the PMOS-MOSCAP `varactor` primitive: accumulation-mode
    varactors use a real cap_var_{lvt,hvt} subckt that models the n-well
    MOSCAP physics directly. The C-V curve has a much wider C_max/C_min
    ratio (4-6x typical, versus the PMOS-MOSCAP's 1.5-2x), which is what
    unlocks LC-VCO tuning ranges in the textbook 15-20% band on a 1.8 V
    supply.

    Pin convention (`p` = "+", `n` = "-") for this primitive:
      * `p` is the gate terminal (c0 in sky130's cap_var subckt). This is
        the high-impedance node where the AC swing happens — in the
        LC-VCO topology the gate is the tank node (clk_out / outn).
      * `n` is the body / n-well terminal (c1 in cap_var, also tied to
        the guard ring `b` pin). DC-biased — in the LC-VCO topology the
        body is the vctrl node.

    This is the OPPOSITE pin assignment of `_varactor` (PMOS-MOSCAP),
    where the gate is the vctrl input and the body is the tank node.
    The flip is intentional: the accumulation varactor's C-V curve is
    parameterised by v(c0) - v(c1), so wiring the AC-driven tank node
    to c0 (gate) gives the tank exposure to the C(Vg-Vb) excursion
    while keeping the DC vctrl reference stable on the body side. The
    PMOS-MOSCAP curve in the existing varactor primitive is shallower
    and tolerates either orientation; the accumulation device's SOA
    range is tighter (|v(c0,c1)| ≤ 2.07 V on sky130 cap_var_lvt) and
    requires the wiring that minimises peak excursion.

    Args:
        flavor: "lvt" (default) → `sky130_fd_pr__cap_var_lvt` — steeper
            C-V slope, higher C_max/C_min ratio, larger tuning range.
            "hvt" → `sky130_fd_pr__cap_var_hvt` — shallower slope, more
            uniform Kvco across the swing, better PLL linearity.
    """
    if flavor not in ("lvt", "hvt"):
        raise ValueError(
            f"_varactor_accum: flavor must be 'lvt' or 'hvt', got "
            f"{flavor!r}. The sky130 PDK exposes both cap_var_lvt and "
            f"cap_var_hvt accumulation-mode N-well varactor subckts."
        )
    cls = "varactor_accum_lvt" if flavor == "lvt" else "varactor_accum_hvt"
    return Device(
        instance_name=name,
        device_class=cls,
        pdk_variant=None,
        pins={"p": p, "n": n},
        sizing={"W": W, "L": L, "M": float(M)},
    )


def build_lc_tank_vco(
    cell_name: str,
    spec: Spec,
    *,
    # ─── Tank reactive elements ─────────────────────────────
    # Full differential-mode tank inductance and capacitance. The
    # inductance is split into two L_tank/2 elements internally
    # (center-tapped at vdd) to keep the topology truly differential.
    L_tank: float = 10e-9,   # 10 nH
    C_tank: float = 63e-12,  # 63 pF → f ≈ 200 MHz with L=10 nH
    # Optional series-loss resistor modelling tank Q. Default 5 Ω is
    # a Q ≈ 12 tank at 200 MHz with L = 10 nH; set higher (lower Q)
    # to make startup harder, lower (higher Q) for cleaner oscillation.
    R_tank_series: float = 5.0,

    # ─── Cross-coupled NMOS pair sizing ─────────────────────
    W_osc: float = 20e-6,    # 20 µm — wide for big gm at moderate I
    L_osc: float = 0.5e-6,   # 0.5 µm — minimum-ish for high f_T

    # ─── Tail current source sizing ─────────────────────────
    W_tail: float = 40e-6,   # 2× W_osc for a 2I-mirror-style tail
    L_tail: float = 0.5e-6,

    # ─── Bias current target ────────────────────────────────
    # Nominal tail current (each cross-coupled MOSFET carries I_bias/2).
    # Not used for sizing in v0.0.56 — kept as a forward-compat knob for
    # when a varactor / current-mirror bias chain lands.
    I_bias: float = 1e-3,    # 1 mA (each half-pair sees 0.5 mA)

    # ─── Tail current source M multiplier (v0.0.59) ─────────
    # Parallel-finger multiplier on Mtail. Defaults to 1 (single finger
    # at W_tail width). Bumping M_tail keeps the tail comfortably on
    # at low vctrl (needed for the varactor-tuned LC-VCO where vctrl
    # must swing low to push the varactor into its high-cap state).
    M_tail: int = 1,

    # ─── Varactor frequency tuning (v0.0.58) ────────────────
    # When `with_varactor=True`, two PMOS-MOSCAP varactors are added in
    # parallel with the tank cap. Each varactor's gate is tied to vctrl;
    # its body / drain-source node connects to one differential tank
    # output (outp or outn). Voltage on vctrl modulates the varactor
    # cap (~Cox*W_var*L_var when |Vgs|>Vth_p, ~0 when off), which adds
    # to the fixed C_tank to give a vctrl-controlled total capacitance.
    # The result: vctrl tunes f_osc = 1/(2π·sqrt(L·(C_tank + 2*C_var))).
    #
    # Tuning math (v0.0.59 honest scope on sky130):
    #   On sky130 BSIM4, a single PMOS-MOSCAP at W=80 µm L=0.5 µm gives
    #   cap range ≈ 14 fF (depletion) to 26 fF (inversion) across Vg
    #   from 1.6V down to 0V. With body at vdd=1.8V, the workable vctrl
    #   range is [0.9V, 1.7V] (vctrl below 0.9V → tail NMOS turns off
    #   → no oscillation). On that workable swing the varactor moves
    #   from depletion (small cap) to weak inversion (larger cap).
    #
    #   For meaningful frequency tuning, the varactor cap delta must
    #   be a significant fraction of total tank cap. At the default
    #   C_tank = 63 pF, the varactor contribution is < 0.1% — flat
    #   frequency response.  To unlock ≥5% tuning, callers must drop
    #   C_tank to ~100-200 fF (target frequency ~2 GHz instead of
    #   200 MHz) AND increase M_var (parallel-finger multiplier) to
    #   8-16. With (C_tank=100 fF, W_var=80 µm, M_var=16, M_tail=4)
    #   on sky130, the measured tuning ratio is ≈ 5.4% across vctrl
    #   ∈ [0.9V, 1.6V], with vctrl up → freq down (PMOS varactor in
    #   inversion at low vctrl → large cap → low freq).
    #
    #   gf180mcu's 3.3V supply gives wider vctrl headroom and can
    #   support deeper inversion → larger tuning ratio. Not validated
    #   end-to-end in v0.0.59; left for a future portability item.
    with_varactor: bool = False,
    W_var: float = 20e-6,    # 20 µm — moderate cap, easy startup
    L_var: float = 0.5e-6,   # 0.5 µm — minimum for high Q varactor
    M_var: int = 1,          # parallel-finger multiplier for the
                              # varactor; bump to 8-16 for meaningful
                              # tuning in conjunction with low C_tank.

    # ─── Multi-varactor parallel topology (v0.0.60) ──────────
    # On gf180mcu, the pmos_3p3 subckt's per-instance W is binned with
    # wmax = 100 µm (sm141064 model card). For sky130, the equivalent
    # max W is 100 µm on pfet_01v8. Once the per-cell varactor needs to
    # be ≥ 100 µm to push the cap delta into a meaningful fraction of
    # the tank, callers hit the W_max wall. Putting N varactors in
    # parallel — each at W=W_var, all four pins identically connected
    # to (vctrl, outp/outn) — gives an effective gate width of
    # N_var_parallel × W_var without exceeding the per-cell bin limit.
    #
    # SPICE-level: each parallel varactor emits as its own xCvar_p<k> /
    # xCvar_n<k> line with the same pin tuple; ngspice adds the
    # subckt capacitances naturally. Equivalent to a single
    # M_var=N_var_parallel × M_var device on PDKs that accept the
    # larger width, but bin-safe everywhere.
    #
    # Defaults to 1 for sky130 back-compat (v0.0.59 sky130 sizing has
    # W_var=80 µm ≤ wmax). Set to 4 on gf180mcu to spread an effective
    # 80 µm gate across 4 × 20 µm devices.
    N_var_parallel: int = 1,

    # ─── vbias_tail decoupling (v0.0.60) ─────────────────────
    # In v0.0.59 the cross-coupled LC-VCO ties Mtail.gate to the same
    # vctrl net the varactors gate on. That couples the tail current
    # to the varactor sweep: vctrl below ~0.9 V on sky130 turns Mtail
    # off (Vgs_tail = vctrl - vss = vctrl < Vth_n ≈ 0.5 V leaves no
    # overdrive to pull the tail into saturation), killing oscillation
    # entirely. The workable vctrl swing collapses to [0.9 V, 1.7 V]
    # on sky130 / [1.2 V, 2.5 V] on gf180mcu — far short of the full
    # [vss, vdd] swing the loop filter would like to deliver.
    #
    # When `decouple_vbias_tail=True`, the cell exposes a NEW PORT
    # `vbias_tail` that is wired to Mtail.gate, leaving vctrl free to
    # drive only the varactors. External callers (testbench, PLL bias
    # chain) hold vbias_tail at a fixed bias point (~0.9 V on sky130,
    # ~1.5 V on gf180mcu) that keeps Mtail comfortably in saturation
    # across the entire vctrl swing. vctrl can now sweep the full
    # [vss, vdd] without the tail-off cliff.
    #
    # Backwards-compatible behaviour (default False):
    #   * Port list stays ['clk_out', 'vctrl', 'vdd', 'vss']
    #   * Mtail.gate = vctrl (v0.0.59 behaviour preserved)
    #
    # When True:
    #   * Port list becomes ['clk_out', 'vctrl', 'vbias_tail', 'vdd',
    #     'vss']
    #   * Mtail.gate = vbias_tail
    #   * Testbench builder reads cell.ports to detect the extra port
    #     and wires a separate VBIAS_TAIL DC source.
    #
    # Expected impact: full-swing vctrl unlocks ≥ 15% tuning ratio on
    # sky130 (v0.0.59 measured 5.0% across the [0.9, 1.7] workable
    # swing; extending to [0, 1.8] roughly triples the C-V excursion).
    decouple_vbias_tail: bool = False,

    # ─── Accumulation-mode N-well varactor (v0.0.91) ─────────
    # The PMOS-MOSCAP varactor (with_varactor=True above) sweeps cap
    # across (Vgs - Vth_p), a 0.6-0.8 V range that gives ~1.5-2x
    # C_max/C_min ratio on sky130's 1.8 V supply. That ceiling cap-ed
    # the v0.0.60 LC-VCO tuning at 9.66% across the full vctrl swing,
    # short of the textbook 15-20% target. The accumulation-mode N-well
    # varactor (sky130_fd_pr__cap_var_{lvt,hvt}) uses a real n-well
    # MOSCAP model whose C(Vg-Vb) curve sweeps from depletion (Cgs/Cgd
    # parasitic floor) through inversion (Cox·W·L) with a 4-6x ratio.
    # Combined with the existing decouple_vbias_tail = 1 architecture
    # (so vctrl swings the full [0, 1.8 V] rail), this primitive
    # delivers measured tuning ratios in the 18-30 % band — well above
    # the 15 % literature target.
    #
    # Wiring (DIFFERENT from `_varactor` PMOS-MOSCAP):
    #   * Gate (c0)  ← tank node (clk_out / outn). The AC swing
    #     drives the gate; v(gate) - v(body) traces the C-V curve as
    #     the tank rings.
    #   * Body (c1)  ← vctrl. DC-biased; vctrl shifts the average
    #     operating point through the C-V curve.
    #
    # SOA caveat: the cap_var_* subckt's internal Q(V) expression uses
    # `cosh((v - vt)/0.15)` and overflows when |v(c0,c1)| > ~2 V. The
    # LC-VCO sizing therefore targets tank peak-to-peak swings ≤ 1.5 V
    # (≤ 750 mV either side of vdd/2). This means lossier R_tank_series
    # and smaller W_osc than the v0.0.60 PMOS-MOSCAP defaults — we trade
    # raw amplitude for a cleaner cap-modulation sweep.
    #
    # When `with_accumulation_varactor=True`, the PMOS-MOSCAP varactor
    # ports above are NOT added. The two are mutually exclusive — if
    # both are True, the accumulation variant wins and the PMOS-MOSCAP
    # is skipped.
    with_accumulation_varactor: bool = False,
    # Flavour selector for the accumulation-mode varactor:
    #   accum_flavor_hvt = 0 (default) → LVT (steeper C-V → bigger
    #                                     tuning range)
    #   accum_flavor_hvt = 1            → HVT (smoother C-V → better
    #                                     Kvco linearity)
    # A numeric flag is used (rather than a string) because the .clay
    # sizing block grammar only accepts numeric values.
    accum_flavor_hvt: bool = False,
    W_var_accum: float = 10e-6,        # 10 µm — moderate per-finger
                                       # width for cap_var_*. Smaller
                                       # than the PMOS-MOSCAP default
                                       # because cap_var has much higher
                                       # cap-per-W (~10× the PMOS).
    L_var_accum: float = 0.5e-6,       # minimum-ish for high Q
    M_var_accum: int = 8,              # parallel-finger multiplier
                                       # (maps to `vm=` on cap_var).
                                       # Default 8 gives ~30-300 fF
                                       # cap swing per varactor.

    # ─── PDK selector ──────────────────────────────────────
    pdk_id: str = "sky130",

    # ─── R120 C: optional qualitative DRC annotations ───────
    # When True, attach three layout-prompt annotations to the LC-VCO
    # cell. Cross-coupled NMOS oscillator topologies are extremely
    # symmetry-sensitive — any imbalance between the two half-circuits
    # injects 1/f noise from the tail into the differential output as
    # AM-to-PM conversion, raising phase noise (Razavi RF §8.7). The
    # three annotations capture:
    #   * MATCHING (Mosc1, Mosc2) — cross-coupled NMOS pair Vth
    #     mismatch directly translates to differential offset on the
    #     tank, modulating the oscillation amplitude and seeding even-
    #     order phase-noise sidebands.
    #   * PASSIVE_MATCHING (L1, L2) — the split inductor halves must
    #     have identical Q and identical inductance value. Asymmetric L
    #     tilts the differential resonance and the cross-coupled pair
    #     compensates with a quiescent imbalance that dumps tail noise
    #     onto the tank.
    #   * GUARD_RING (Mtail,) — the tail NMOS is the dominant 1/f
    #     noise source feeding flicker noise into the tank common-mode.
    #     A pwell guard ring tied to vss isolates Mtail from neighbor
    #     digital substrate noise, which would otherwise couple into
    #     the LC-VCO output through Mtail's bulk-source capacitance.
    # Annotations live only in cell.__dict__["_drc_annotations"];
    # SPICE emit is byte-for-byte unaffected when default
    # attach_drc=False.
    attach_drc: bool = False,
) -> Cell:
    """Build a Cell IR for the cross-coupled NMOS LC-tank VCO.

    Topology
    --------
    Differential LC-tank VCO with cross-coupled NMOS pair providing
    negative resistance and an NMOS tail current source gated by
    `vctrl`. The tank is a parallel-LC across the differential output
    pair (outp / outn); the inductor is split as two L_tank/2 halves
    center-tapped at vdd. A small series-loss resistor (R_tank_series)
    models the finite Q of the inductor.

    Why a single-ended `clk_out` port
    ---------------------------------
    Compatibility with the v0.0.40+ PLL plumbing — the ring VCO's
    surface port list is `(vctrl, clk_out, vdd, vss)`, and the PLL
    testbench wrapper looks for a port literally named `clk_out`. The
    LC-VCO routes its differential output `outp` to the `clk_out` port;
    the opposite phase `outn` is an internal net. A future fully-
    differential PLL wrapper can expose both phases as outer ports.

    Args:
        cell_name: name for the Cell IR.
        spec: target spec (may be empty for compile-only verification).
        L_tank: full differential tank inductance in henries. Split into
            two L_tank/2 elements internally.
        C_tank: full differential tank capacitance in farads.
        R_tank_series: lumped series loss in ohms modelling inductor Q.
        W_osc, L_osc: cross-coupled NMOS pair sizing.
        W_tail, L_tail: tail current source NMOS sizing.
        I_bias: nominal tail current (informational; not used to size
            the tail mirror in this minimal v0.0.56 build).
        pdk_id: target PDK string written into the Cell IR. Drivers
            consult this to pick the right device subckt bindings.

    Returns:
        A Cell IR with the LC-tank topology materialized, ports
        ['clk_out', 'vctrl', 'vdd', 'vss'].
    """
    # Port list — single-ended clk_out for PLL compatibility.
    # When decouple_vbias_tail=True, `vbias_tail` is appended as an
    # extra port (positioned before vdd/vss to keep supplies last,
    # matching the testbench's expectation order).
    if decouple_vbias_tail:
        ports = ["clk_out", "vctrl", "vbias_tail", "vdd", "vss"]
    else:
        ports = ["clk_out", "vctrl", "vdd", "vss"]
    devices: list[Device] = []

    # Internal nets:
    #   outp / outn  — differential tank outputs. outp == clk_out.
    #   mid_p, mid_n — center-tap nodes between each inductor half and
    #                  its series-loss resistor.
    #   tail         — common source of the cross-coupled NMOS pair.
    #
    # We alias outp to clk_out so the port name appears directly in
    # device pin maps (rather than going through an extra wire device).
    outp = "clk_out"
    outn = "outn"

    # ─── Tank inductors (split as L_tank/2 each, center-tapped at vdd) ──
    # Path: vdd → R_loss_p → mid_p → L1 → outp.  Mirror on the n-side.
    # R_loss_p / R_loss_n model the finite Q of each inductor half.
    half_L = L_tank / 2.0
    devices.append(_ind("L1", p="mid_p", n=outp, value=half_L))
    devices.append(_ind("L2", p="mid_n", n=outn, value=half_L))

    # Tank-loss series resistors (split to keep the bridge symmetric).
    half_R = R_tank_series / 2.0
    devices.append(Device(
        instance_name="R_loss_p",
        device_class="res",
        pdk_variant=None,
        pins={"p": "vdd", "n": "mid_p"},
        sizing={"value": half_R},
    ))
    devices.append(Device(
        instance_name="R_loss_n",
        device_class="res",
        pdk_variant=None,
        pins={"p": "vdd", "n": "mid_n"},
        sizing={"value": half_R},
    ))

    # ─── Tank capacitor (full differential value across outp / outn) ───
    devices.append(_cap("C_tank", p=outp, n=outn, value=C_tank))

    # ─── Varactor frequency tuning (v0.0.58, opt-in) ──────────────────
    # Two PMOS MOSCAP varactors, one on each differential output. Each
    # has gate=vctrl, body=output_node (drain+source tied to body in
    # the SPICE emit pattern). The differential effect: both caps
    # modulate together with vctrl, adding to C_tank. Single-ended
    # cap modulation would unbalance the tank; symmetric loading
    # preserves differential symmetry while tuning frequency.
    #
    # v0.0.60: when N_var_parallel > 1, emit N independent varactor
    # devices on each tank node instead of one. SPICE-level the parallel
    # devices add naturally; the workaround sidesteps the per-cell W_max
    # bin limit (100 µm on gf180mcu's pmos_3p3, same on sky130
    # pfet_01v8) when callers want effective gate widths > 100 µm.
    # Skip the PMOS-MOSCAP path when the accumulation-mode variant is
    # active — they are mutually exclusive. Mixing them on the same tank
    # would over-load the cap budget and the calibrated sizing here
    # assumes one or the other, not both.
    if with_varactor and not with_accumulation_varactor:
        n_parallel = max(int(N_var_parallel), 1)
        for k in range(n_parallel):
            # Suffix only when n_parallel > 1 so the default case
            # preserves the v0.0.59 instance names (Cvar_p / Cvar_n)
            # — important for back-compat with any test or doc that
            # references the canonical names.
            if n_parallel == 1:
                suffix = ""
            else:
                suffix = f"_{k}"
            devices.append(_varactor(
                f"Cvar_p{suffix}", p="vctrl", n=outp,
                W=W_var, L=L_var, M=M_var,
            ))
            devices.append(_varactor(
                f"Cvar_n{suffix}", p="vctrl", n=outn,
                W=W_var, L=L_var, M=M_var,
            ))

    # ─── Accumulation-mode N-well varactor frequency tuning (v0.0.91) ──
    # Wiring is inverted relative to the PMOS-MOSCAP path: the tank node
    # (clk_out / outn) drives the GATE (c0), and vctrl drives the BODY
    # (c1). See `_varactor_accum` docstring for the rationale (cap_var
    # subckt SOA + cap-vs-(c0-c1) parameterisation).
    #
    # The C_max/C_min ratio of cap_var_lvt at sky130 is ~5.7×; cap_var_hvt
    # is ~4.3×. Either flavour gives >15% LC-VCO tuning at the sizing
    # below — LVT is the default for maximum tuning range, HVT is
    # available for callers who need smoother Kvco linearity.
    if with_accumulation_varactor:
        flavor = "hvt" if accum_flavor_hvt else "lvt"
        devices.append(_varactor_accum(
            "Cvar_accum_p", p=outp, n="vctrl",
            W=W_var_accum, L=L_var_accum, M=M_var_accum,
            flavor=flavor,
        ))
        devices.append(_varactor_accum(
            "Cvar_accum_n", p=outn, n="vctrl",
            W=W_var_accum, L=L_var_accum, M=M_var_accum,
            flavor=flavor,
        ))

    # ─── Cross-coupled NMOS pair (negative resistance generator) ──────
    # Mosc1: drain=outp, gate=outn (cross), source=tail, body=vss
    devices.append(_mosfet(
        "Mosc1", "nmos",
        d=outp, g=outn, s="tail", b="vss",
        W=W_osc, L=L_osc, M=1,
    ))
    # Mosc2: drain=outn, gate=outp (cross), source=tail, body=vss
    devices.append(_mosfet(
        "Mosc2", "nmos",
        d=outn, g=outp, s="tail", b="vss",
        W=W_osc, L=L_osc, M=1,
    ))

    # ─── Tail current source (NMOS) ─────────────────────────────────
    # Default (v0.0.59 behaviour): gate = vctrl. Higher vctrl → more
    # tail current → better startup margin AND higher f_osc through
    # the gm-dependent device capacitance.
    #
    # v0.0.60 decouple_vbias_tail=True: gate = vbias_tail. External
    # bias holds the tail current constant while vctrl drives only
    # the varactors. Unlocks full [vss, vdd] vctrl swing for tuning.
    tail_gate = "vbias_tail" if decouple_vbias_tail else "vctrl"
    devices.append(_mosfet(
        "Mtail", "nmos",
        d="tail", g=tail_gate, s="vss", b="vss",
        W=W_tail, L=L_tail, M=M_tail,
    ))

    cell = Cell(
        name=cell_name,
        ports=ports,
        spec=spec,
        devices=devices,
        pdk_id=pdk_id,
    )

    # ─── R120 C: optional DRC annotations ───────────────────
    # MATCHING (Mosc1, Mosc2) / PASSIVE_MATCHING (L1, L2) /
    # GUARD_RING (Mtail). See the kwarg docs block above for the
    # per-annotation rationale.
    if attach_drc:
        from ..pdks import (
            DrcAnnotation,
            DrcKind,
            attach_drc_annotation,
        )

        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.MATCHING,
            devices=("Mosc1", "Mosc2"),
            extras={
                "rationale": (
                    "Cross-coupled NMOS oscillator pair — Vth mismatch "
                    "between Mosc1 and Mosc2 translates directly to a "
                    "differential offset on the LC tank, modulating "
                    "amplitude and seeding even-order phase-noise "
                    "sidebands via AM-to-PM conversion (Razavi RF §8.7)"
                ),
                "ratio": "1:1",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.PASSIVE_MATCHING,
            devices=("L1", "L2"),
            extras={
                "rationale": (
                    "Split LC-tank inductor halves L_tank/2 — the "
                    "differential resonance assumes L1 and L2 have "
                    "identical inductance AND identical Q. Asymmetric "
                    "inductors tilt the tank centre frequency between "
                    "the two phases, forcing the cross-coupled pair "
                    "into a quiescent imbalance that dumps tail 1/f "
                    "noise onto the differential output"
                ),
                "primitive": "symmetric_spiral_inductor",
            },
        ))
        attach_drc_annotation(cell, DrcAnnotation(
            kind=DrcKind.GUARD_RING,
            devices=("Mtail",),
            extras={
                "rationale": (
                    "Tail NMOS is the dominant 1/f noise source feeding "
                    "flicker noise into the LC tank common-mode; pwell "
                    "guard ring tied to vss isolates Mtail from "
                    "neighbour digital substrate noise that would "
                    "otherwise couple into the VCO output through "
                    "Mtail's bulk-source capacitance and re-modulate "
                    "the oscillation frequency"
                ),
                "well_type": "pwell",
            },
        ))

    return cell
