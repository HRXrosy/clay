"""
v0.0.69 Phase 3 third slice — Adjoint Network local descent adapter.

Plugs the clay_ir Adjoint Network gradient extractor into the optimizer
harness as a **local-convergence** step that runs AFTER a global
explorer (tSS-BO) has proposed a candidate sizing.

The hybrid pipeline:

  1. tSS-BO (Stage 1) proposes K candidate sizings via its truncated
     subspace surrogate. Cheap exploration of the ~10-dim sizing space.
  2. For each candidate, the harness runs ngspice once to evaluate
     "the truth" (gain, PM, GBW, CMRR, ...). tSS-BO updates its prior.
  3. For the small-signal targets (Gain, CMRR), the adjoint local
     refiner takes each candidate and runs a few gradient-descent
     steps using the analytical ∂Gain/∂W vector — much faster than
     the next ngspice round, since the gradient is computed in the
     Rust adjoint without invoking the simulator.
  4. The best refined candidate is fed back to the next tSS-BO step
     as a "high-quality seed" so the BO posterior tightens around the
     local optimum.

This module ships the local-refiner side only — the BO orchestration
stays in `tssbo_adapter.py`. The two compose via the
`AdjointLocalRefiner` class, which has a single public method
`refine(sizing, cell_builder, target_gain) -> Sizing`.

Honest scope (matches the `clay/_ir_bridge.compute_gradients` slice):

  * Only DC gain is differentiable today — PM / GBW gradients arrive
    in v0.0.70+ once the small-signal model gains the capacitive
    entries.
  * Only W is a differentiable knob (not L, not Cc, not bias currents).
  * The local refiner is "local-OP": it assumes the DC operating point
    is stable across W perturbations. A full outer fixpoint is v0.0.70+.

The adapter degrades gracefully when `clay_ir` is unavailable — the
`refine` method becomes a no-op pass-through, so the optimizer harness
can still run with or without the Rust extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


# ─── Configuration ───────────────────────────────────────────────


@dataclass(frozen=True)
class AdjointLocalConfig:
    """Tunable parameters for the local descent loop.

    Defaults are conservative — meant to refine, not to globally search.
    The whole point of pairing with tSS-BO is to let the global explorer
    handle multi-mode hunting; the adjoint refiner only polishes the
    last 1-2 orders of magnitude of gain shortfall.

    Fields:
      * ``max_steps`` — gradient-descent iterations per refine() call.
        4 is enough for the local quadratic basin near a typical opamp
        bias point; 8+ is bordering on "tSS-BO would have done that
        cheaper".
      * ``step_size`` — multiplier applied to the gradient before adding
        to W. Honest: the gradient is in [d(gain) / d(W in meters)] so
        a 1e-9 scaling lands W bumps in nanometers; we expose the knob
        so the caller can match it to the cell's W magnitude.
      * ``min_w`` / ``max_w`` — clamp W to a physically plausible
        window. Defaults match sky130's nfet_01v8 W envelope; gf180mcu
        callers should override.
      * ``in_node`` / ``out_node`` — net names the adjoint method
        differentiates between. Defaults match the cell-template
        convention (`vin` / `vout`).
      * ``supply_vdd`` / ``pdk_id`` — forwarded to the Rust adjoint.
    """

    max_steps: int = 4
    step_size: float = 1e-9
    min_w: float = 0.3e-6
    max_w: float = 100e-6
    in_node: str = "vin"
    out_node: str = "vout"
    supply_vdd: float = 1.8
    pdk_id: str = "sky130"


# ─── Refinement record (for logging / debug) ─────────────────────


@dataclass
class AdjointRefineTrace:
    """Per-step trace of one `refine()` call. Optional — caller can
    enable via `AdjointLocalRefiner(record_trace=True)`. The harness
    consumes this for the CLI table; tests inspect specific entries.
    """

    initial_sizing: dict[str, float] = field(default_factory=dict)
    final_sizing: dict[str, float] = field(default_factory=dict)
    initial_gain: float = float("nan")
    final_gain: float = float("nan")
    steps: list[dict[str, Any]] = field(default_factory=list)
    aborted: bool = False
    abort_reason: Optional[str] = None


# ─── Public class ────────────────────────────────────────────────


class AdjointLocalRefiner:
    """Local gradient-descent refiner that wraps the Rust adjoint.

    Usage::

        refiner = AdjointLocalRefiner(cell_builder=build_my_cell)
        refined = refiner.refine(sizing_from_bo)

    The `cell_builder` is the same callable the harness uses — it
    consumes the sizing dict (after the refiner finishes touching it)
    and returns a `clay.ir.Cell`. The refiner constructs the Rust
    handle internally on each step so changes to W feed back into the
    next gradient computation.

    The refiner only touches knobs whose name starts with `W_` (the
    template convention). Other knobs (e.g. `gm_id_diff`, `Cc`,
    `I_ref`) pass through unchanged.
    """

    def __init__(
        self,
        cell_builder: Callable[[Mapping[str, float]], Any],
        *,
        config: Optional[AdjointLocalConfig] = None,
        record_trace: bool = False,
    ) -> None:
        self.cell_builder = cell_builder
        self.config = config or AdjointLocalConfig()
        self.record_trace = record_trace
        # Lazily populated on first refine() call so a missing clay_ir
        # extension doesn't raise at construction time.
        self._bridge_module = None
        self._has_bridge: Optional[bool] = None

    def _load_bridge(self) -> bool:
        if self._has_bridge is not None:
            return self._has_bridge
        try:
            from clay import _ir_bridge as bridge

            self._bridge_module = bridge
            self._has_bridge = bridge.has_rust_ir()
        except ImportError:
            self._has_bridge = False
        return self._has_bridge

    def refine(
        self,
        sizing: Mapping[str, float],
    ) -> tuple[dict[str, float], Optional[AdjointRefineTrace]]:
        """Run up to `config.max_steps` local descent steps on the W
        knobs in `sizing`. Returns the refined sizing and an optional
        trace. The original mapping is not mutated.

        Falls back to a pass-through (returns a shallow copy of the
        input) when:
          * `clay_ir` is not importable, OR
          * the cell_builder raises on the initial sizing, OR
          * no W knob is present in the sizing (nothing to refine).

        These graceful fallbacks let the harness keep running even on
        machines / cells where the adjoint isn't applicable. Trace is
        None when `record_trace=False`.
        """
        out: dict[str, float] = dict(sizing)
        trace: Optional[AdjointRefineTrace]
        if self.record_trace:
            trace = AdjointRefineTrace(
                initial_sizing=dict(sizing),
                final_sizing=dict(sizing),
            )
        else:
            trace = None

        # Identify W knobs. The template convention is `W_xxx`; we also
        # accept the rare bare "W" for single-device test cells.
        w_knobs = [k for k in out.keys() if k == "W" or k.startswith("W_")]
        if not w_knobs:
            if trace is not None:
                trace.aborted = True
                trace.abort_reason = "no W knob present"
            return out, trace

        if not self._load_bridge():
            if trace is not None:
                trace.aborted = True
                trace.abort_reason = "clay_ir extension not available"
            return out, trace
        bridge = self._bridge_module
        assert bridge is not None  # type-narrowing for mypy

        # Build the initial cell + initial gain. If either fails, bail
        # out gracefully — the optimizer harness still has the unrefined
        # sizing in hand.
        try:
            cell = self.cell_builder(out)
        except Exception as exc:  # noqa: BLE001
            if trace is not None:
                trace.aborted = True
                trace.abort_reason = f"cell_builder raised: {exc!r}"
            return out, trace

        initial_gain = bridge.compute_dc_gain(
            cell,
            self.config.in_node,
            self.config.out_node,
            supply_vdd=self.config.supply_vdd,
            pdk_id=self.config.pdk_id,
        )
        if initial_gain is None or not math.isfinite(initial_gain):
            if trace is not None:
                trace.aborted = True
                trace.abort_reason = f"initial gain not finite: {initial_gain}"
            return out, trace
        if trace is not None:
            trace.initial_gain = float(initial_gain)

        # The bridge takes `device_instance_name` strings; the template
        # convention is that knob name `W_X` maps to device `MX` (the
        # parser strips the leading "W_" / capitalises). For a generic
        # refiner we can't assume a 1:1 mapping — instead we let the
        # cell builder pick the instance names and identify them by
        # walking the cell.
        #
        # The builder can return either a `clay.ir.Cell` (Python IR) or
        # a `clay_ir.CellGraphHandle` (Rust handle). Detect via duck-
        # typing on the `devices` attribute (Cell has it, handle doesn't).
        device_names: list[str] = []
        if hasattr(cell, "devices"):
            # Python Cell path.
            from clay.ir import Device  # local import — cheap when unused

            device_names = [
                d.instance_name
                for d in cell.devices
                if isinstance(d, Device)
                and d.device_class in ("nmos", "pmos")
            ]
        else:
            # CellGraphHandle path. Walk the dense node id space and
            # ask the handle for each kind / name. The CellGraphHandle
            # uses a monotonic id counter so 0..node_count covers all
            # device nodes (NMOS/PMOS plus any passives we'll filter
            # out by node_kind).
            n_nodes = cell.node_count()
            for nid in range(n_nodes):
                try:
                    kind = cell.node_kind(nid)
                except Exception:  # noqa: BLE001
                    continue
                if kind in ("nmos", "pmos"):
                    try:
                        device_names.append(cell.node_instance_name(nid))
                    except Exception:  # noqa: BLE001
                        continue

        # Greedy step: at each iter, compute the gradient w.r.t. every
        # MOS instance's W and walk against it by a small step. The
        # sign convention follows the gradient (we want gain MAGNITUDE
        # to increase — note that for an inverting amp the "gain" the
        # adjoint returns is negative, so we step away from zero).
        # Real callers should set `step_size` so a single step doesn't
        # bound off the local quadratic basin.
        current_gain = float(initial_gain)

        for step_idx in range(self.config.max_steps):
            try:
                grads = bridge.compute_gradients(
                    cell,
                    self.config.in_node,
                    self.config.out_node,
                    device_names,
                    supply_vdd=self.config.supply_vdd,
                    pdk_id=self.config.pdk_id,
                )
            except Exception as exc:  # noqa: BLE001
                if trace is not None:
                    trace.aborted = True
                    trace.abort_reason = f"compute_gradients raised: {exc!r}"
                break
            if grads is None or not grads:
                if trace is not None:
                    trace.aborted = True
                    trace.abort_reason = "no gradients returned"
                break

            # Step direction: move W against the gradient if the current
            # gain is positive (we want it bigger); move WITH the
            # gradient if it's negative (we want |gain| bigger ⇒ make it
            # MORE negative ⇒ subtract |grad| * sign(current_gain)).
            # The simplest robust rule: maximize |gain| by stepping in
            # +sign(grad·gain) direction so the chain rule of
            # d|gain|/dW = sign(gain) · d(gain)/dW is always positive.
            sign_of_gain = 1.0 if current_gain >= 0 else -1.0
            step_record: dict[str, Any] = {
                "step": step_idx,
                "gain_before": current_gain,
                "deltas": {},
            }

            # Apply per-knob updates. Map device_name → W_knob_name by
            # the template convention (W_<rest> matches the device name
            # whose `instance_name` is "M" + rest, or by a more general
            # heuristic: take the trailing chars of the device's
            # instance_name and look for a W_<that> in the sizing).
            applied_any = False
            for dev_name, grad in grads.items():
                # Pick the W knob: first try a direct alphabetical match
                # — many templates name their knob "W_<DeviceName>" but
                # some use shorter aliases. We probe a small set.
                candidate_keys = [
                    f"W_{dev_name}",
                    f"W_{dev_name.lstrip('M')}",
                    f"W_{dev_name.lower()}",
                    f"W_{dev_name.lstrip('M').lower()}",
                ]
                key = next(
                    (k for k in candidate_keys if k in out),
                    None,
                )
                if key is None:
                    continue
                # delta_W = step_size * sign(gain) * grad
                # so that d|gain|/dW · delta_W > 0 (always improving).
                delta = self.config.step_size * sign_of_gain * grad
                new_w = out[key] + delta
                # Clamp to the configured envelope so a runaway step
                # doesn't violate PDK W limits.
                new_w_clamped = min(
                    self.config.max_w, max(self.config.min_w, new_w)
                )
                if new_w_clamped != out[key]:
                    out[key] = new_w_clamped
                    step_record["deltas"][key] = float(
                        new_w_clamped - (out[key] - delta)
                    )
                    applied_any = True

            if not applied_any:
                if trace is not None:
                    trace.aborted = True
                    trace.abort_reason = (
                        f"no W knob matched device list at step {step_idx}"
                    )
                break

            # Re-build the cell + recompute the gain with updated W.
            try:
                cell = self.cell_builder(out)
            except Exception as exc:  # noqa: BLE001
                if trace is not None:
                    trace.aborted = True
                    trace.abort_reason = (
                        f"cell_builder raised on refined sizing at step "
                        f"{step_idx}: {exc!r}"
                    )
                break
            new_gain = bridge.compute_dc_gain(
                cell,
                self.config.in_node,
                self.config.out_node,
                supply_vdd=self.config.supply_vdd,
                pdk_id=self.config.pdk_id,
            )
            if new_gain is None or not math.isfinite(new_gain):
                if trace is not None:
                    trace.aborted = True
                    trace.abort_reason = (
                        f"refined gain not finite at step {step_idx}: {new_gain}"
                    )
                break
            step_record["gain_after"] = float(new_gain)
            if trace is not None:
                trace.steps.append(step_record)
            current_gain = float(new_gain)

        if trace is not None:
            trace.final_sizing = dict(out)
            trace.final_gain = current_gain
        return out, trace


# ─── Hybrid pipeline orchestrator (documentation entry) ───────────


def hybrid_pipeline_refine_candidates(
    candidates: Sequence[Mapping[str, float]],
    cell_builder: Callable[[Mapping[str, float]], Any],
    *,
    config: Optional[AdjointLocalConfig] = None,
) -> list[dict[str, float]]:
    """Apply the Adjoint local refiner to a batch of tSS-BO candidates.

    Use this as the "Stage 2 polish" between the BO's `propose()` batch
    and the harness's ngspice round. tSS-BO proposes K candidates with
    its truncated subspace surrogate; each candidate gets up to
    `config.max_steps` of adjoint descent; the harness then simulates
    the K refined candidates with ngspice and feeds the truth back to
    the BO posterior.

    Returns the list of refined sizings in the same order as `candidates`.
    Each candidate is independent — failures on one (cell builder
    raised, gain non-finite, etc.) don't affect the others; we just
    pass through the original unrefined sizing for the failure.

    The wider hybrid pipeline lives in the harness's run loop (the BO
    orchestration in `tssbo_adapter.py` doesn't know about the adjoint
    refiner); this function is the documented composition point and
    a useful unit-test surface.
    """
    refiner = AdjointLocalRefiner(cell_builder, config=config)
    out: list[dict[str, float]] = []
    for sizing in candidates:
        refined, _trace = refiner.refine(sizing)
        out.append(refined)
    return out
