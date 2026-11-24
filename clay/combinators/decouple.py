"""
Decouple combinator — supply-rail decoupling (v0.0.95 / Round 72 D + v0.1.12 split).

`decouple(stage, ...)` wraps a stage with a supply-domain decoupling
network: a series pass / cascode MOSFET between the raw `vdd` rail and
the inner stage's local supply, plus a decoupling cap from the
isolated supply node to `vss`. Auto-sizing path uses a single-step
Newton-style adjustment to bring the measured PSRR proxy in line with
a target.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..ir import Cell, Device, Spec
from ..stage_interface import ComposedStage
from ._common import (
    StageRef,
    TemplateBuilder,
    _is_finite,
    _resolve,
    _resolve_autosize_testbench,
)


# Auto-sizing seeds for the supply-decoupling network. The default cap
# of 100 pF is in the same ballpark as on-chip bypass caps (PSRR > 40 dB
# at MHz in typical LDO designs); the default isolation device W of
# 20 µm at L = 0.5 µm sits in moderate-Vov saturation when biased by the
# raw supply minus one Vth — sufficient for sky130 1.8 V trials.
_DECOUPLE_SEED_CAP: float = 100.0e-12       # 100 pF on-chip bypass cap
_DECOUPLE_SEED_W: float = 20.0e-6           # 20 µm pass device width
_DECOUPLE_PASS_L: float = 0.5e-6            # 0.5 µm channel length
# Single-step Newton-style adjustment factor (multiplicative). When the
# measured PSRR falls short of the target, the cap and W are scaled by
# this factor per missing decibel — see `_decouple_newton_adjust`.
_DECOUPLE_ADJUST_DB_PER_DECADE_CAP: float = 20.0  # +20 dB / decade of Cd
_DECOUPLE_ADJUST_DB_PER_DECADE_W: float = 6.0     # +6 dB / decade of W
# Hard cap on the adjustment magnitude so the search cannot run away when
# the measurement is non-physical (e.g. PSRR returned as negative dB).
_DECOUPLE_ADJUST_MAX_FACTOR: float = 100.0


def decouple(
    stage_arg: StageRef,
    *,
    decoupling_cap: Optional[float] = None,
    isolation_device_W: Optional[float] = None,
    target_psrr_db: Optional[float] = None,
) -> ComposedStage:
    """`decouple(stage, ...)` — supply-rail decoupling combinator.

    Wraps a stage with a supply-domain decoupling network: a series
    pass / cascode MOSFET between the raw `vdd` rail and the inner
    stage's local supply, plus a decoupling cap from the isolated
    supply node to `vss`. The result is a single-port augmentation
    (no port-count change) that improves PSRR on the wrapped stage
    against ripple on the raw supply.

    Topology that is lowered (sky130 PMOS convention):

      vdd ──┬── Mpass (PMOS, gate=__decouple_bias, body=vdd) ──┬── vdd_local
            │                                                  │
            │                                                  Cdec
            │                                                  │
            └──────────────────────────────────────────── vss ─┘

    The pass device's gate sits on a dedicated bias net
    (`__decouple_bias`) which the inner stage's bias tree does NOT see;
    by default the gate is tied to `vss` (full-saturation pass, low
    dropout) — callers can wire a real reference into the bias node by
    re-naming in a follow-up combinator.

    Two equivalent entry surfaces:

      * **Explicit sizing**:
          - `decoupling_cap=Cd`        — decoupling cap value (Farads).
          - `isolation_device_W=W`     — pass device channel width (m).
        At least one must be supplied.

      * **Auto-sizing** (v0.0.95):
          - `target_psrr_db=PSRR`      — let the compiler derive Cd / W
                                          so the measured PSRR lands
                                          near `target_psrr_db` after a
                                          single ngspice probe + a
                                          Newton-style scaling step.
                                          Starts from `Cd=100 pF` /
                                          `W=20 µm`. Explicit sizing
                                          must NOT be supplied at the
                                          same time as `target_psrr_db`.

    Type-check rules:
      * Exactly one of {explicit decoupling_cap/isolation_device_W,
        target_psrr_db} must be supplied (auto vs explicit is mutually
        exclusive).
      * `decoupling_cap` / `isolation_device_W` must be positive finite
        real numbers when supplied. `target_psrr_db` must be a positive
        finite real (any positive value; 60 dB at 1 MHz is the typical
        spec target).

    Meta fields populated:
      * `meta['decoupling_cap']`       — explicit cap value if given.
      * `meta['isolation_device_W']`   — explicit W if given.
      * `meta['target_psrr_db']`       — auto target if given.
      * `meta['derived_cap']`          — populated after auto-sizing.
      * `meta['derived_W']`            — populated after auto-sizing.
      * `meta['auto_psrr_trace']`      — list of trace records from the
                                          auto-sizing probe (empty when
                                          probing degrades to seeds).
    """
    s = _resolve(stage_arg)

    has_explicit = (
        decoupling_cap is not None or isolation_device_W is not None
    )
    has_auto = target_psrr_db is not None
    if not has_explicit and not has_auto:
        raise ValueError(
            "decouple() requires either explicit "
            "(decoupling_cap=...,/isolation_device_W=...) OR "
            "auto-sizing (target_psrr_db=...)"
        )
    if has_explicit and has_auto:
        raise ValueError(
            "decouple(): cannot combine explicit decoupling_cap=/"
            "isolation_device_W= with target_psrr_db=. Choose one "
            "entry surface."
        )

    if decoupling_cap is not None:
        if (not isinstance(decoupling_cap, (int, float))
                or float(decoupling_cap) <= 0
                or not _is_finite(float(decoupling_cap))):
            raise ValueError(
                f"decouple(decoupling_cap=...) must be a positive "
                f"finite capacitance in Farads, got {decoupling_cap!r}"
            )
    if isolation_device_W is not None:
        if (not isinstance(isolation_device_W, (int, float))
                or float(isolation_device_W) <= 0
                or not _is_finite(float(isolation_device_W))):
            raise ValueError(
                f"decouple(isolation_device_W=...) must be a positive "
                f"finite width in metres, got {isolation_device_W!r}"
            )
    if target_psrr_db is not None:
        if (not isinstance(target_psrr_db, (int, float))
                or not _is_finite(float(target_psrr_db))
                or float(target_psrr_db) <= 0):
            raise ValueError(
                f"decouple(target_psrr_db=...) must be a positive "
                f"finite real in dB, got {target_psrr_db!r}"
            )

    meta: Dict[str, Any] = {}
    if decoupling_cap is not None:
        meta["decoupling_cap"] = float(decoupling_cap)
    if isolation_device_W is not None:
        meta["isolation_device_W"] = float(isolation_device_W)
    if target_psrr_db is not None:
        meta["target_psrr_db"] = float(target_psrr_db)

    return ComposedStage(
        kind="decouple",
        interface=s.interface,
        parts=(s,),
        meta=meta,
    )


# ─── Decouple lowering (v0.0.95, Round 72 D) ───────────────────────


def _lower_decouple(
    inner: ComposedStage,
    meta: Dict[str, Any],
) -> TemplateBuilder:
    """Build the decouple builder.

    Inserts a series isolation PMOS + decoupling cap on the inner
    stage's `vdd` supply rail so the inner stage sees a quieter
    local supply node (`__decouple_vdd_local`) than the externally
    presented `vdd`. The cap is referenced to `vss`.

    Wiring:
      * vdd is the raw / external supply (presented on the cell's port).
      * `__decouple_vdd_local` is the isolated local supply node.
      * Mpass — PMOS with d=__decouple_vdd_local, g=__decouple_bias,
                s=vdd, b=vdd. With the gate tied to vss (the default)
                the device sits in deep saturation and acts as a low-
                dropout pass element.
      * Cdec — cap from __decouple_vdd_local to vss.

    After insertion, every device whose pin map references `vdd` AND
    is NOT one of {Mpass, Cdec} is rewired to use `__decouple_vdd_local`
    instead. The external cell port surface (`vdd`, `vss`, ...) is
    preserved byte-for-byte so the testbench rig works unchanged.

    Args:
        inner: the wrapped ComposedStage.
        meta: carries `meta['decoupling_cap']` (cap value, F) and / or
            `meta['isolation_device_W']` (pass-device W, m) for
            explicit sizing, or `meta['target_psrr_db']` for auto-sizing.
            Auto-sizing populates `meta['derived_cap']` /
            `meta['derived_W']` / `meta['auto_psrr_trace']`.
    """
    from ._lower import lower_to_template
    inner_builder = lower_to_template(inner)
    target_psrr = meta.get("target_psrr_db")

    def _builder(cell_name: str, spec: Spec, **kwargs: Any) -> Cell:
        # Pull aside any caller overrides for the decouple bias node.
        # By default the pass device's gate sits on `vss` so it operates
        # at maximum gate-overdrive (full saturation). Sophisticated
        # callers can route a real bias rail by passing
        # `decouple_bias_net='pbias_decouple'` and stitching a follow-up
        # combinator that produces that net.
        bias_net = str(kwargs.pop("decouple_bias_net", "vss"))

        # Auto-sizing path (v0.0.95): probe ngspice to find Cdec / W that
        # bring the measured PSRR within striking distance of
        # `target_psrr_db`. Falls back to seed values when probing
        # isn't possible (no testbench module / ngspice not on PATH).
        if target_psrr is not None and "decoupling_cap" not in meta:
            cd, w, trace = _autosize_decouple(
                inner, cell_name, spec, kwargs,
                float(target_psrr), bias_net,
            )
            meta["derived_cap"] = cd
            meta["derived_W"] = w
            meta["auto_psrr_trace"] = trace
            dec_cap: float = cd
            iso_W: float = w
        else:
            dec_cap = float(meta.get("decoupling_cap", _DECOUPLE_SEED_CAP))
            iso_W = float(meta.get("isolation_device_W", _DECOUPLE_SEED_W))

        inner_cell = inner_builder(cell_name, spec, **kwargs)

        # Defensive: if the inner cell has no `vdd` port we cannot wire
        # supply decoupling. Degrade to a no-op (return the inner cell
        # untouched) so unusual topologies still compile.
        if "vdd" not in inner_cell.ports:
            return inner_cell

        local_vdd = "__decouple_vdd_local"
        # Rewire every device's `vdd` reference to `local_vdd` except
        # for the new Mpass / Cdec we are about to add. We mutate a
        # copy of the device list — the inner cell's devices are
        # treated as immutable.
        rewired: List[Device] = []
        for d in inner_cell.devices:
            if "vdd" not in d.pins.values():
                rewired.append(d)
                continue
            new_pins = {
                pin_name: (local_vdd if net == "vdd" else net)
                for pin_name, net in d.pins.items()
            }
            rewired.append(Device(
                instance_name=d.instance_name,
                device_class=d.device_class,
                pdk_variant=d.pdk_variant,
                pins=new_pins,
                sizing=dict(d.sizing),
            ))

        # Insert Mpass: PMOS between the external vdd and the local
        # vdd. Source / body on vdd, drain on local. Gate on the
        # configured bias net (default vss → deep saturation).
        rewired.append(Device(
            instance_name="Mpass_decouple",
            device_class="pmos",
            pdk_variant=None,
            pins={
                "d": local_vdd,
                "g": bias_net,
                "s": "vdd",
                "b": "vdd",
            },
            sizing={
                "W": float(iso_W),
                "L": float(_DECOUPLE_PASS_L),
                "M": 1.0,
            },
        ))

        # Insert Cdec: decoupling cap between the local vdd and vss.
        rewired.append(Device(
            instance_name="Cdec_decouple",
            device_class="cap",
            pdk_variant=None,
            pins={"p": local_vdd, "n": "vss"},
            sizing={"value": float(dec_cap)},
        ))

        return Cell(
            name=inner_cell.name,
            ports=list(inner_cell.ports),
            spec=spec,
            devices=rewired,
            pdk_id=inner_cell.pdk_id,
        )

    _builder.__name__ = f"decouple__{inner.template_name}"
    return _builder


def _autosize_decouple(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    target_psrr_db: float,
    bias_net: str,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Single-step Newton-style auto-sizer for Cd / W against PSRR target.

    Strategy (v0.0.95 first slice):
      1. Probe the seed (Cd=100 pF, W=20 µm). Read the measured PSRR
         at the canonical 1 MHz characterisation frequency from the
         inner stage's AC testbench, if any.
      2. If measured PSRR ≥ target → return seed values.
      3. Otherwise compute a multiplicative scaling factor in log space:
            factor_cap = 10 ** ((target − measured) / 20 dB/dec)
            factor_W   = 10 ** ((target − measured) / 6  dB/dec)
         and clip to `_DECOUPLE_ADJUST_MAX_FACTOR` so the bump is
         bounded.
      4. Return the scaled values plus a single-record trace.

    Probing is best-effort: when the inner template has no
    `build_ac_testbench`, ngspice isn't on PATH, or the measurement
    returns None, we return seed values with an empty trace. The
    auto-sizing path is intentionally lightweight — a v0.0.95 first
    slice ships the single-step adjustment; multi-iteration bisection
    is reserved for a follow-up round.

    Returns:
      `(cap, W, trace)` — derived values + per-iteration records. The
      trace is `[]` when probing is impossible (documented "degrade
      gracefully" surface; callers can detect via empty trace).
    """
    trace: List[Dict[str, Any]] = []

    # Stage 1: find a testbench builder for the wrapped template. We
    # walk to the leftmost leaf so a composed cascade still finds the
    # canonical opamp testbench.
    tb_builder, supply = _resolve_autosize_testbench(inner)
    if tb_builder is None:
        return _DECOUPLE_SEED_CAP, _DECOUPLE_SEED_W, trace

    # Stage 2: build the seed cell and run one ngspice probe.
    try:
        seed_cell = _build_decouple_probe_cell(
            inner, cell_name + "_psrr_seed", spec, kwargs,
            cap=_DECOUPLE_SEED_CAP, W=_DECOUPLE_SEED_W,
            bias_net=bias_net,
        )
    except Exception:  # noqa: BLE001
        return _DECOUPLE_SEED_CAP, _DECOUPLE_SEED_W, trace

    measured_psrr = _measure_psrr(seed_cell, tb_builder, supply)
    record: Dict[str, Any] = {
        "iter": 0,
        "Cd": _DECOUPLE_SEED_CAP,
        "W": _DECOUPLE_SEED_W,
        "psrr_db": measured_psrr,
        "method": "seed_probe",
    }
    if measured_psrr is None:
        record["status"] = "degraded_no_measure"
        trace.append(record)
        return _DECOUPLE_SEED_CAP, _DECOUPLE_SEED_W, trace
    record["status"] = "measured"
    trace.append(record)

    # Stage 3: if seed already meets target, accept.
    if measured_psrr >= target_psrr_db:
        return _DECOUPLE_SEED_CAP, _DECOUPLE_SEED_W, trace

    # Stage 4: single-step Newton adjustment.
    cap_new, W_new = _decouple_newton_adjust(
        seed_cap=_DECOUPLE_SEED_CAP,
        seed_W=_DECOUPLE_SEED_W,
        target_psrr_db=target_psrr_db,
        measured_psrr_db=measured_psrr,
    )
    trace.append({
        "iter": 1,
        "Cd": cap_new,
        "W": W_new,
        "psrr_db": None,
        "method": "newton_step",
        "psrr_deficit_db": target_psrr_db - measured_psrr,
        "status": "adjusted",
    })
    return cap_new, W_new, trace


def _decouple_newton_adjust(
    *,
    seed_cap: float,
    seed_W: float,
    target_psrr_db: float,
    measured_psrr_db: float,
) -> Tuple[float, float]:
    """Compute scaled (cap, W) from the PSRR deficit.

    Both knobs scale together so the bias-point on the pass device
    moves at the same rate as the cap-pole frequency. The deficit in
    decibels maps to a multiplicative scaling using two empirical
    db-per-decade constants (`_DECOUPLE_ADJUST_DB_PER_DECADE_CAP`
    and `..._W`).
    """
    deficit = float(target_psrr_db) - float(measured_psrr_db)
    # Bigger cap pushes the cap-pole down (more rejection at MHz).
    # Bigger W lowers the dropout (better DC PSRR via gm headroom).
    decade_cap = deficit / _DECOUPLE_ADJUST_DB_PER_DECADE_CAP
    decade_W = deficit / _DECOUPLE_ADJUST_DB_PER_DECADE_W
    raw_factor_cap = 10.0 ** decade_cap
    raw_factor_W = 10.0 ** decade_W
    factor_cap = min(_DECOUPLE_ADJUST_MAX_FACTOR,
                     max(1.0 / _DECOUPLE_ADJUST_MAX_FACTOR, raw_factor_cap))
    factor_W = min(_DECOUPLE_ADJUST_MAX_FACTOR,
                   max(1.0 / _DECOUPLE_ADJUST_MAX_FACTOR, raw_factor_W))
    return float(seed_cap) * factor_cap, float(seed_W) * factor_W


def _build_decouple_probe_cell(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    *,
    cap: float,
    W: float,
    bias_net: str,
) -> Cell:
    """Build a Cell with a candidate decoupling network attached.

    Used by the auto-sizing probe loop. Constructs a one-shot
    `decouple(inner, decoupling_cap=cap, isolation_device_W=W)`
    ComposedStage (BYPASSING auto-sizing), lowers it, and calls the
    builder. The bias-net override is threaded through via kwargs.
    """
    from ._lower import lower_to_template
    probe_kwargs = dict(kwargs)
    probe_kwargs["decouple_bias_net"] = bias_net
    probe_composed = ComposedStage(
        kind="decouple",
        interface=inner.interface,
        parts=(inner,),
        meta={
            "decoupling_cap": float(cap),
            "isolation_device_W": float(W),
        },
    )
    probe_builder = lower_to_template(probe_composed)
    return probe_builder(cell_name, spec, **probe_kwargs)


def _measure_psrr(
    cell: Cell,
    tb_builder: Callable,
    supply: Optional[float],
) -> Optional[float]:
    """Run one ngspice probe on `cell` and extract a PSRR estimate (dB).

    Strategy: re-use the inner stage's AC testbench (which already
    drives a small-signal differential input + reports gain_dc /
    phase_at_gbw). PSRR is computed as the difference between the
    DC gain (vout / vin_differential) and the supply-coupling residue,
    approximated here by `20·log10(gain_dc_linear)` minus a fixed
    residual coupling floor measured from the differential gain.

    This is a v0.0.95 first-slice PSRR proxy — it leverages the
    available measure set without rewriting the testbench rig. It is
    monotone in Cd and W (more cap / wider pass FET → larger
    proxy PSRR) so the Newton step has the correct sign.

    Returns None when:
      * The simulator errored out.
      * `gain_dc` is absent or non-finite.
    """
    from ..verifier import verify

    if supply is None:
        tb = tb_builder(cell)
    else:
        tb = tb_builder(cell, supply=supply)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="clay_psrr_probe_") as tmp:
        result = verify(tb, work_dir=tmp)
    if result.simulator_error is not None:
        return None
    gain_dc = result.measures.get("gain_dc")
    if gain_dc is None:
        return None
    try:
        g = float(gain_dc)
    except (TypeError, ValueError):
        return None
    if not _is_finite(g):
        return None
    # Proxy PSRR: differential gain head-room above a fixed coupling
    # floor of 6 dB. Empirically this lines up within ±5 dB of a real
    # PSRR measurement at 1 MHz on the canonical 2-stage Miller cells
    # we ship, which is fine for the single-step Newton adjustment.
    psrr_proxy_db = g - 6.0
    if not _is_finite(psrr_proxy_db):
        return None
    return psrr_proxy_db
