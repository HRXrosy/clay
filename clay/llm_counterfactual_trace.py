"""
v0.1.01 Round 78 Agent B — Counterfactual Trace as First-Class LLM Input

What this module ships
----------------------

A new Clay language artifact — the *counterfactual trace* — that turns the
existing adjoint-network gradient surface into a prompt-shaped object the LLM
authoring loop can see.  The trace answers the question the LLM most needs and
the existing scoring path silently hides: *if I bump knob X by one physical
unit, how does spec metric S move?*

Today the bridge's :func:`clay._ir_bridge.compute_gradients` returns a Python
``dict[str, float]`` of ``device_instance_name → ∂metric/∂W`` that flows
straight into the optimiser (tSS-BO, adjoint local refiner).  The LLM never
sees the gradient.  As a result the LLM learns "this design satisfied spec
A" rather than "the design satisfied A and a 1 µm bump on M1's width would
move gain by +2.3 dB and pm by −0.4 deg" — the causal pattern an experienced
analog engineer carries in her head.

This module is the philosophy supplement §5.3 (Counterfactual Trace expose
to LLM) made concrete.  Three public surfaces:

  * :class:`CounterfactualEntry` — one ``(knob, ∂metric, units, spec_metric)``
    record with all the post-decoration metadata the prompt formatter needs.
  * :func:`compute_counterfactual_trace` — runs Clay's existing adjoint API
    over a list of (device, knob_kind) pairs for every spec metric in a
    spec list, returns the merged entries.  No new PyO3 surface.
  * :func:`format_trace_for_llm_prompt` — emits a markdown table that drops
    into the next retry prompt verbatim.  The R79+ hook in
    :mod:`tools.llm_authoring.real_api_run` consumes this.
  * :func:`select_top_k_actionable` — heuristic that filters out small /
    sign-ambiguous / non-finite entries so the prompt budget goes to the
    knobs that actually move the spec.

Honest scope (matches the bridge layer)
---------------------------------------

  * Only knobs the bridge already supports flow through.  Today that's
    ``W`` and ``Vbias`` (`compute_gradients`'s `_ALLOWED_KINDS`).  An
    ``L`` or ``Cc`` knob in the caller's input list passes through the
    bridge's silent-coerce-to-W path, so the trace still has an entry —
    the entry is labelled with its requested kind and the gradient is
    *with respect to* whatever the bridge actually computed.  The
    formatter notes the coercion in the units column so a downstream
    consumer can see the discrepancy at a glance.
  * Spec metrics: anything the bridge accepts via
    ``compute_gradients(..., metric=...)`` — today ``"gain_dc"`` (dB),
    ``"gbw"`` (Hz), ``"pm"`` (degrees).  Any other metric in a Spec yields
    an entry with ``derivative_value=None`` and ``reason="unsupported_metric"``.
  * No simulator round-trips fire — the entire trace is one
    bridge.compute_gradients call per metric over the device list.  The
    cost is one adjoint solve per metric (O(milliseconds) on the
    canonical 2-stage Miller cell).
  * The function is a no-op (returns ``[]``) when the ``clay_ir``
    extension is unavailable (no Rust toolchain build).  Tests guard with
    ``pytest.importorskip("clay_ir")`` exactly like the existing
    adjoint surface.

Backward compatibility
----------------------

NOT wired anywhere by default.  The R78 B mission narrowly adds the
trace module plus an *opt-in* CLI flag on the real-API runner
(``--counterfactual-trace``).  When the flag is off the runner is
byte-identical to R77 D's grammar-aware path.  This preserves:

  * R77 D real_api_run.py grammar-constrained path byte-exact;
  * R78 A `clay/distance_to_sat.py` byte-exact (different file);
  * v0.0.94 R70 A bridge cap normalise / v0.0.95 R71 D mos normalise /
    v0.0.97 R73 B dc_op.rs / v0.0.98 R74 C measure_parse /
    v0.0.99 R75 B testbench_emit byte-exact (the bridge surface this
    module calls is the same `compute_gradients` shipped in v0.0.69+).

Reference
---------

  * Audit ``docs/audits/2026-05-19-clay-llm-extrapolation-philosophy.html``
    §5.3 (Counterfactual Trace expose to LLM) + §9 position 3
    (unclaimed PL × LLM scaffold).
  * Spec ``docs/superpowers/specs/clay-counterfactual-trace-llm-channel.html``.

Module is import-safe even if ``clay_ir`` is missing — the bridge module's
guard layer handles that case downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# ─── Metric registry ────────────────────────────────────────────────


# Maps a spec-metric name (matching `clay/_ir_bridge.compute_gradients`'s
# `metric` kwarg) to its physical unit + a short prompt-friendly tag.
# The tag is what the LLM sees in the rendered trace ("dB", "Hz", "deg").
_METRIC_UNITS: dict[str, tuple[str, str]] = {
    # name (lower) → (rendered_unit, output_unit_prefix)
    "gain_dc": ("V/V", "gain"),
    "gbw": ("Hz", "gbw"),
    "pm": ("deg", "pm"),
}

# Spec keywords downstream code uses interchangeably with the bridge
# metric names.  When a Spec carries one of these, we route it to the
# bridge metric on the right.
_SPEC_ALIAS: dict[str, str] = {
    "gain_dc": "gain_dc",
    "dc_gain": "gain_dc",
    "gain": "gain_dc",
    "gbw": "gbw",
    "ugbw": "gbw",
    "unity_gain_bw": "gbw",
    "ft": "gbw",
    "pm": "pm",
    "phase_margin": "pm",
    "phasemargin": "pm",
}


def _normalize_spec_metric(metric_name: str) -> Optional[str]:
    """Map a spec keyword to the bridge metric name.  Case-insensitive.

    Returns the canonical bridge metric (``"gain_dc"`` / ``"gbw"`` /
    ``"pm"``) when an alias matches, or ``None`` for an unsupported
    metric (no adjoint surface in the v0.0.93 bridge).
    """
    if not isinstance(metric_name, str):
        return None
    key = metric_name.strip().lower()
    return _SPEC_ALIAS.get(key)


# ─── Knob name disambiguation ───────────────────────────────────────


# Knob kinds the bridge knows about today.  See
# `clay/_ir_bridge.compute_gradients` -> `_ALLOWED_KINDS`.
_BRIDGE_KIND_W = "W"
_BRIDGE_KIND_VBIAS = "Vbias"
_BRIDGE_ALLOWED_KINDS = (_BRIDGE_KIND_W, _BRIDGE_KIND_VBIAS)


def _coerce_kind_for_bridge(kind: str) -> tuple[str, bool]:
    """Return ``(coerced_kind, was_coerced)``.

    The bridge silently coerces unknown kinds (``L`` / ``Cc`` / ``M``)
    to ``W`` so older callers don't crash.  This module mirrors that
    behaviour but RECORDS the coercion so the prompt formatter can
    surface the fact in the units column — the LLM should not be misled
    into thinking it received an ``∂metric/∂L`` value when the bridge
    actually returned an ``∂metric/∂W``.
    """
    if kind in _BRIDGE_ALLOWED_KINDS:
        return kind, False
    return _BRIDGE_KIND_W, True


# ─── Core record ────────────────────────────────────────────────────


@dataclass(frozen=True)
class CounterfactualEntry:
    """One row in a counterfactual trace.

    The record is intentionally flat — every field is a primitive that
    serialises to JSON cleanly so the trace can travel through the
    real-API runner's persisted artifact (the ``PhaseRecord`` row).

    Fields
    ------
    param_name
        The device instance name in the Clay cell — e.g. ``"M1"``,
        ``"Mtail"``, ``"Iref"``, ``"Cc"``.  Matches the cell's
        ``Device.instance_name`` exactly.
    knob_kind
        The requested knob kind: ``"W"`` / ``"L"`` / ``"Cc"`` / ``"M"`` /
        ``"Vbias"``.  Echoed back even when the bridge coerces (see
        :attr:`bridge_kind`).
    bridge_kind
        What kind the bridge actually used.  Equal to :attr:`knob_kind`
        when the kind is in ``_BRIDGE_ALLOWED_KINDS``, otherwise ``"W"``
        and :attr:`was_coerced` is True.
    derivative_value
        The signed gradient ``∂spec_metric/∂param`` in *bridge units*
        (W in m, Vbias in A — the cur_src ``value`` is a current). The
        LLM sees a rescaled rendering through :meth:`render_value` so
        the units stay human-friendly (per-µm for W, per-µA for Vbias).
        ``None`` when the bridge could not compute the gradient (e.g.
        unsupported metric, missing instance, no Rust extension).
    units
        Bridge-side physical unit string: e.g. ``"V/V per m"`` for
        ``∂gain_dc/∂W``, ``"deg per F"`` for ``∂pm/∂Cc``.  The prompt
        formatter renders a rescaled column so the LLM sees nicer
        magnitudes.
    spec_metric
        The bridge metric name this gradient belongs to: ``"gain_dc"``,
        ``"gbw"``, ``"pm"``.  Echoed verbatim so a downstream consumer
        can group by metric without parsing :attr:`units`.
    was_coerced
        True when :attr:`knob_kind` did not match a bridge-supported
        kind and the bridge silently used ``"W"`` instead.
    reason
        Short tag describing why a gradient is missing.  Set to ``"ok"``
        on success; otherwise one of ``"unsupported_metric"``,
        ``"clay_ir_missing"``, ``"compute_failed"``,
        ``"instance_not_found"``.
    """

    param_name: str
    knob_kind: str
    bridge_kind: str
    derivative_value: Optional[float]
    units: str
    spec_metric: str
    was_coerced: bool = False
    reason: str = "ok"

    # ── Convenience accessors for the formatter ──

    def is_actionable(self) -> bool:
        """Return True iff the entry carries a finite, non-zero gradient.

        The retry-prompt formatter uses this to drop entries that
        wouldn't help the LLM (e.g. NaN, ±0.0, or any of the missing-
        gradient reasons).
        """
        if self.derivative_value is None:
            return False
        v = float(self.derivative_value)
        if not math.isfinite(v):
            return False
        return v != 0.0

    def render_value(self) -> str:
        """Render :attr:`derivative_value` as a human-friendly number.

        Empty string when the value is missing.  The formatter rescales
        the magnitude so a per-meter gradient becomes per-µm, a per-Farad
        gradient becomes per-pF, etc.  See :func:`_rescaled_value_and_unit`.
        """
        if self.derivative_value is None:
            return ""
        rescaled, _ = _rescaled_value_and_unit(
            float(self.derivative_value),
            self.bridge_kind,
            self.spec_metric,
        )
        if rescaled is None:
            return f"{float(self.derivative_value):+.3g}"
        return f"{rescaled:+.3g}"

    def render_unit(self) -> str:
        """Render the unit string at the rescaled magnitude.

        Falls back to :attr:`units` (the raw bridge unit) when the
        rescaler does not have a rule for the (knob_kind, spec_metric)
        pair.  Always returns a non-empty string when :attr:`reason` is
        ``"ok"``.
        """
        rescaled, unit_label = _rescaled_value_and_unit(
            float(self.derivative_value) if self.derivative_value is not None else 0.0,
            self.bridge_kind,
            self.spec_metric,
        )
        if unit_label is None:
            return self.units
        return unit_label


# ─── Rescaling so the LLM sees readable magnitudes ──────────────────


def _rescaled_value_and_unit(
    value: float,
    bridge_kind: str,
    spec_metric: str,
) -> tuple[Optional[float], Optional[str]]:
    """Convert a per-SI-unit gradient into a per-engineering-unit one.

    Examples (illustrative, sky130-scale magnitudes):

        ∂gain_dc/∂W      = 5.0e+4 V/V/m   →   5.0e-2 V/V/µm
        ∂gbw/∂W          = 2.0e+11 Hz/m   →   2.0e+5  Hz/µm
        ∂pm/∂W           = 1.0e+5 deg/m   →   1.0e-1 deg/µm
        ∂pm/∂Cc          = 5.0e+13 deg/F  →   5.0e+1 deg/pF
        ∂gain_dc/∂Vbias  = 1.0e+5 V/V/A   →   1.0e-1 V/V/µA

    Returns ``(rescaled_value, unit_label)``.  ``unit_label`` is None
    when no rescaling rule applies — the caller should fall back to the
    raw unit string.
    """
    if not math.isfinite(value):
        return None, None
    if bridge_kind == _BRIDGE_KIND_W:
        # W is in metres; engineering scale is µm.  Divide by 1e6.
        scale = 1e-6
        unit_label = _per_unit_label(spec_metric, "µm")
        return value * scale, unit_label
    if bridge_kind == _BRIDGE_KIND_VBIAS:
        # Vbias is a current (cur_src.value); engineering scale is µA.
        scale = 1e-6
        unit_label = _per_unit_label(spec_metric, "µA")
        return value * scale, unit_label
    if bridge_kind == "Cc":
        # Cc is in farads; engineering scale is pF.  Divide by 1e12.
        scale = 1e-12
        unit_label = _per_unit_label(spec_metric, "pF")
        return value * scale, unit_label
    if bridge_kind == "L":
        scale = 1e-6
        unit_label = _per_unit_label(spec_metric, "µm")
        return value * scale, unit_label
    return None, None


def _per_unit_label(spec_metric: str, engineering_unit: str) -> str:
    """Return e.g. ``"V/V per µm"`` / ``"Hz per pF"`` / ``"deg per µA"``.

    Falls back to ``"<metric> per <unit>"`` for unknown metrics so the
    label never collapses to an empty string in the trace.
    """
    info = _METRIC_UNITS.get(spec_metric.lower())
    metric_unit = info[0] if info is not None else spec_metric
    return f"{metric_unit} per {engineering_unit}"


# ─── compute_counterfactual_trace ───────────────────────────────────


def _resolve_metric_list(
    spec_list: Iterable[Any],
) -> list[tuple[str, str]]:
    """Walk a spec list and return ``(canonical_metric, original_name)``.

    Accepts a variety of shapes so callers don't need to massage their
    spec objects:

      * A list of strings — each treated as a metric name.
      * A list of ``(metric_name, target_value)`` tuples — only the
        name is used; target value is irrelevant to the gradient.
      * A list of objects with a ``metric`` attribute (e.g.
        :class:`clay.ir.SpecTarget`).
      * A dict mapping metric name → anything.

    Duplicate metric names (after normalisation) are de-duplicated; the
    first occurrence wins.  Unsupported metrics are kept with
    canonical=None so the caller can emit an "unsupported_metric"
    entry.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    if spec_list is None:
        return out
    if isinstance(spec_list, dict):
        iterable: Iterable[Any] = list(spec_list.keys())
    else:
        iterable = spec_list
    for item in iterable:
        if isinstance(item, str):
            name = item
        elif isinstance(item, tuple) and len(item) >= 1:
            name = str(item[0])
        elif hasattr(item, "metric"):
            name = str(getattr(item, "metric"))
        elif hasattr(item, "name"):
            name = str(getattr(item, "name"))
        else:
            # Unknown shape — coerce to string.
            name = str(item)
        canonical = _normalize_spec_metric(name)
        key = (canonical or name).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((canonical or "", name))
    return out


def _normalize_param_list(
    params: Iterable[Any],
) -> list[tuple[str, str]]:
    """Normalise ``params`` to ``[(instance_name, knob_kind), ...]``.

    Accepts:
      * ``str`` — kind defaults to ``"W"``.
      * ``(str, str)`` — explicit ``(instance_name, knob_kind)``.

    Other shapes raise ``TypeError`` with a useful message.  Duplicates
    are preserved (the bridge accepts duplicate names — last write wins
    in its result dict — and the LLM-side formatter handles dedup at
    render time).
    """
    out: list[tuple[str, str]] = []
    for entry in params:
        if isinstance(entry, str):
            out.append((entry, _BRIDGE_KIND_W))
        elif isinstance(entry, tuple) and len(entry) == 2:
            inst, kind = entry
            out.append((str(inst), str(kind)))
        else:
            raise TypeError(
                f"counterfactual trace param entry must be str or "
                f"(str, str), got {type(entry).__name__}: {entry!r}"
            )
    return out


def compute_counterfactual_trace(
    cell_or_handle: Any,
    spec_list: Iterable[Any],
    params: Iterable[Any],
    *,
    in_node: str = "vin",
    out_node: str = "vout",
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    top_k: Optional[int] = 8,
) -> list[CounterfactualEntry]:
    """Compute the counterfactual trace for one (cell, spec_list, params).

    Walks the bridge's :func:`compute_gradients` once per metric and
    flattens the results into a list of :class:`CounterfactualEntry`.

    Parameters
    ----------
    cell_or_handle
        A :class:`clay.ir.Cell` or a pre-built
        :class:`clay_ir.CellGraphHandle`.  Same calling contract as
        :func:`clay._ir_bridge.compute_gradients` — the bridge accepts
        both.
    spec_list
        Iterable of spec-metric specifiers.  See
        :func:`_resolve_metric_list` for the accepted shapes.
    params
        Iterable of ``str`` instance names or ``(instance_name,
        knob_kind)`` tuples.  See :func:`_normalize_param_list`.
    in_node / out_node / supply_vdd / pdk_id
        Forwarded verbatim to :func:`compute_gradients`.  Defaults
        match the v0.0.69 stacked-NMOS test fixture conventions
        (``"vin"`` / ``"vout"``).
    top_k
        Cap the returned list size after the actionable-entries filter.
        ``None`` to disable the cap (return every entry, even
        unsupported / non-actionable ones).  Default 8 — the §5.3
        worked example top entry budget.  When the cap is hit, the
        non-actionable entries are dropped *first*; once everything
        actionable fits, the cap stops trimming.

    Returns
    -------
    list[CounterfactualEntry]
        One entry per ``(param, metric)`` pair, sorted (per metric) by
        descending absolute gradient.  Empty list when the ``clay_ir``
        extension is unavailable.
    """
    # Lazy import the bridge so a missing clay_ir extension doesn't
    # blow up at module-load time.  Mirrors the AdjointLocalRefiner
    # pattern used in tools/clay_optimize/adjoint_local.py.
    try:
        from clay import _ir_bridge as bridge
    except ImportError:
        return []
    if not getattr(bridge, "_HAS_CLAY_IR", False):
        return []

    metric_list = _resolve_metric_list(spec_list)
    param_list = _normalize_param_list(params)
    if not metric_list or not param_list:
        return []

    # Build a single bridge-shaped param list.  The bridge accepts
    # mixed (instance_name, kind) tuples; we deliberately do NOT
    # collapse Cc / L / M to W ourselves — let the bridge's coercion
    # path do it so the result list shape matches what the bridge
    # already returns elsewhere.
    bridge_params: list[tuple[str, str]] = []
    coerced_for_bridge: dict[str, tuple[str, bool]] = {}
    for inst, kind in param_list:
        coerced_kind, was_coerced = _coerce_kind_for_bridge(kind)
        bridge_params.append((inst, coerced_kind))
        # Last-write-wins matches dict semantics — duplicates would
        # collide on the bridge return anyway.
        coerced_for_bridge[inst] = (coerced_kind, was_coerced)

    entries: list[CounterfactualEntry] = []
    for canonical_metric, original_name in metric_list:
        if not canonical_metric:
            # Unsupported metric — emit one row per param so the LLM
            # sees the metric and the reason explicitly.
            for inst, kind in param_list:
                coerced_kind, was_coerced = coerced_for_bridge.get(
                    inst, (_BRIDGE_KIND_W, False)
                )
                entries.append(
                    CounterfactualEntry(
                        param_name=inst,
                        knob_kind=kind,
                        bridge_kind=coerced_kind,
                        derivative_value=None,
                        units=_per_unit_label(original_name, _engineering_unit_for_kind(coerced_kind)),
                        spec_metric=original_name,
                        was_coerced=was_coerced,
                        reason="unsupported_metric",
                    )
                )
            continue
        # Bridge-supported metric: one call returns the whole dict.
        try:
            grad_dict = bridge.compute_gradients(
                cell_or_handle,
                in_node,
                out_node,
                bridge_params,
                supply_vdd=supply_vdd,
                pdk_id=pdk_id,
                metric=canonical_metric,
            )
        except Exception:
            grad_dict = None
        if grad_dict is None:
            # clay_ir not built, or the bridge swallowed an error — emit
            # one row per param with reason="compute_failed".
            for inst, kind in param_list:
                coerced_kind, was_coerced = coerced_for_bridge.get(
                    inst, (_BRIDGE_KIND_W, False)
                )
                entries.append(
                    CounterfactualEntry(
                        param_name=inst,
                        knob_kind=kind,
                        bridge_kind=coerced_kind,
                        derivative_value=None,
                        units=_per_unit_label(canonical_metric, _engineering_unit_for_kind(coerced_kind)),
                        spec_metric=canonical_metric,
                        was_coerced=was_coerced,
                        reason="compute_failed",
                    )
                )
            continue
        # Map back from instance_name → gradient.  The bridge returns
        # device_instance_name keys; entries the bridge couldn't resolve
        # to a node id come back as 0.0 (the sentinel path documented in
        # `compute_gradients`).  We surface those with
        # reason="instance_not_found" so the LLM doesn't waste a slot
        # on a fake-zero.  The disambiguation rule: a 0.0 from a name
        # that was never returned (key missing) is `instance_not_found`;
        # a 0.0 from a returned key is "ok" with the obvious gradient
        # of zero (e.g. a sub-threshold device).  The bridge today
        # always returns every requested key, so absence is rare —
        # but we keep the path robust.
        for inst, kind in param_list:
            coerced_kind, was_coerced = coerced_for_bridge.get(
                inst, (_BRIDGE_KIND_W, False)
            )
            raw_unit = _per_unit_label(canonical_metric, _engineering_unit_for_kind(coerced_kind))
            if inst not in grad_dict:
                entries.append(
                    CounterfactualEntry(
                        param_name=inst,
                        knob_kind=kind,
                        bridge_kind=coerced_kind,
                        derivative_value=None,
                        units=raw_unit,
                        spec_metric=canonical_metric,
                        was_coerced=was_coerced,
                        reason="instance_not_found",
                    )
                )
                continue
            v = grad_dict[inst]
            if v is None or not math.isfinite(float(v)):
                entries.append(
                    CounterfactualEntry(
                        param_name=inst,
                        knob_kind=kind,
                        bridge_kind=coerced_kind,
                        derivative_value=None,
                        units=raw_unit,
                        spec_metric=canonical_metric,
                        was_coerced=was_coerced,
                        reason="compute_failed",
                    )
                )
                continue
            entries.append(
                CounterfactualEntry(
                    param_name=inst,
                    knob_kind=kind,
                    bridge_kind=coerced_kind,
                    derivative_value=float(v),
                    units=raw_unit,
                    spec_metric=canonical_metric,
                    was_coerced=was_coerced,
                    reason="ok",
                )
            )

    # Sort per metric by |gradient| descending so the LLM sees the most
    # impactful knob first.  Stable sort across metrics — entries from
    # the first metric in `metric_list` come first.
    by_metric: dict[str, list[CounterfactualEntry]] = {}
    metric_order: list[str] = []
    for e in entries:
        key = e.spec_metric
        if key not in by_metric:
            by_metric[key] = []
            metric_order.append(key)
        by_metric[key].append(e)
    flat: list[CounterfactualEntry] = []
    for m in metric_order:
        bucket = by_metric[m]
        bucket.sort(
            key=lambda e: (
                # actionable rows first
                0 if e.is_actionable() else 1,
                # then larger |value| first
                -abs(float(e.derivative_value))
                if e.derivative_value is not None
                else float("inf"),
            )
        )
        flat.extend(bucket)

    # Apply the top_k cap.  Strategy: drop non-actionable rows first
    # (preserves the most informative entries for the LLM), then
    # truncate the tail.
    if top_k is None or top_k >= len(flat):
        return flat
    actionable = [e for e in flat if e.is_actionable()]
    non_actionable = [e for e in flat if not e.is_actionable()]
    if len(actionable) >= top_k:
        return actionable[:top_k]
    # Mix in non-actionable entries up to the cap to keep some signal
    # about what was unsupported / missing.
    remaining = top_k - len(actionable)
    return actionable + non_actionable[:remaining]


def _engineering_unit_for_kind(bridge_kind: str) -> str:
    """Return the engineering-scale unit for a bridge knob kind.

    Used by the raw-units label so a non-actionable / non-rescaled row
    still carries a unit string in the trace.  ``W`` → µm, ``Vbias`` →
    µA, ``Cc`` → pF, ``L`` → µm, others → "unit".
    """
    if bridge_kind == _BRIDGE_KIND_W or bridge_kind == "L":
        return "µm"
    if bridge_kind == _BRIDGE_KIND_VBIAS:
        return "µA"
    if bridge_kind == "Cc":
        return "pF"
    return "unit"


# ─── select_top_k_actionable ────────────────────────────────────────


def select_top_k_actionable(
    entries: Iterable[CounterfactualEntry],
    *,
    top_k: int = 8,
    abs_floor: float = 0.0,
) -> list[CounterfactualEntry]:
    """Filter ``entries`` to the top-K actionable rows.

    Heuristic dropouts (in order):
      1. ``reason != "ok"`` — non-finite, missing instance, unsupported
         metric.
      2. ``derivative_value`` not finite or exactly 0.0.
      3. ``abs(derivative_value) < abs_floor`` — caller-set magnitude
         floor.  Default 0.0 (no floor); pass e.g. 1.0 to drop sub-unit
         entries on a ∂PM/∂Cc trace.

    The remaining rows are re-sorted by descending ``|derivative_value|``
    (this is the LLM's "best signal first" ordering) and truncated to
    ``top_k``.

    Returns a new list — never mutates the input.
    """
    out: list[CounterfactualEntry] = []
    for e in entries:
        if e.reason != "ok":
            continue
        if not e.is_actionable():
            continue
        if e.derivative_value is None:
            continue
        if abs(float(e.derivative_value)) < abs_floor:
            continue
        out.append(e)
    out.sort(
        key=lambda e: -abs(float(e.derivative_value or 0.0)),
    )
    if top_k <= 0:
        return []
    return out[:top_k]


# ─── v0.1.05 Round 81 Agent C: distance × counterfactual composition ─


# The three legal weight_mode strings the composition function understands.
# See :func:`select_top_k_by_distance` for the per-mode semantics.
WEIGHT_MODE_DISTANCE: str = "distance"
WEIGHT_MODE_NORMALIZED: str = "normalized"
WEIGHT_MODE_LINEAR: str = "linear"

# Public allowlist used by the entry-point validators.  Kept module-level
# so a downstream consumer (e.g. a CLI flag) can introspect the legal
# weight_modes without copying the string literals.
_WEIGHT_MODES: tuple[str, ...] = (
    WEIGHT_MODE_DISTANCE,
    WEIGHT_MODE_NORMALIZED,
    WEIGHT_MODE_LINEAR,
)


def _build_distance_map(
    violations: Iterable[Any],
) -> dict[str, float]:
    """Return ``{spec_name: distance}`` keyed by the canonical metric.

    Accepts anything that walks like a :class:`SpecViolation` — only the
    ``spec_name`` and ``distance`` attributes are read.  Unsatisfied
    specs contribute their distance; satisfied specs land at ``0.0``.
    Both the original spec name and the canonical-bridge metric name
    appear as keys so a downstream lookup against a counterfactual
    entry's ``spec_metric`` field finds the violation even when the
    spec author wrote ``gain`` and the bridge wrote ``gain_dc``.

    Non-finite distances (``math.inf`` from a missing measure) collapse
    to ``math.inf`` in the map — the caller's weighted-score code
    handles that case by treating the entry as the most urgent
    (max-priority) row in the trace.
    """
    out: dict[str, float] = {}
    for v in violations:
        name = getattr(v, "spec_name", None)
        if name is None:
            continue
        d = getattr(v, "distance", None)
        if d is None:
            continue
        # Original name verbatim (case-preserving so a caller using
        # "Gain_DC" still matches).
        out[str(name)] = float(d) if isinstance(d, (int, float)) else math.inf
        # Case-folded.
        out[str(name).lower()] = out[str(name)]
        # Canonical bridge metric (so an entry tagged "gain_dc" still
        # locates a violation whose spec_name was authored as "gain").
        canonical = _normalize_spec_metric(str(name))
        if canonical is not None:
            out[canonical] = out[str(name)]
    return out


def _lookup_distance(
    distance_map: dict[str, float],
    spec_metric: str,
) -> Optional[float]:
    """Look up an entry's distance.  Tries verbatim, lower, canonical.

    Returns ``None`` when no key matches.  The caller treats ``None``
    as "no distance signal for this metric" (weighted score falls back
    to the bare ``|gradient|`` from :func:`select_top_k_actionable`).
    """
    if spec_metric in distance_map:
        return distance_map[spec_metric]
    key = spec_metric.lower()
    if key in distance_map:
        return distance_map[key]
    canonical = _normalize_spec_metric(spec_metric)
    if canonical is not None and canonical in distance_map:
        return distance_map[canonical]
    return None


def _normalize_distance_for_score(
    distance: Optional[float],
    *,
    inf_replacement: float = 1e9,
) -> float:
    """Project a distance onto the ``[0, +∞)`` weight axis.

    ``None`` → ``0`` (no distance signal); ``math.inf`` → a large
    finite penalty so the score doesn't itself become inf and break
    Python's sort.  Negative distances are clamped to 0 (defensive —
    :class:`SpecViolation.distance` is non-negative by contract).
    """
    if distance is None:
        return 0.0
    if not math.isfinite(distance):
        return inf_replacement
    return max(0.0, float(distance))


def _max_abs_gradient(entries: Iterable[CounterfactualEntry]) -> float:
    """Return the largest ``|derivative_value|`` over the actionable rows.

    Used by ``weight_mode="normalized"`` and ``"linear"`` to keep each
    half of the score on a roughly comparable scale before summing /
    multiplying.  Returns ``0.0`` when no actionable row exists, in
    which case the caller skips the normalisation step.
    """
    best = 0.0
    for e in entries:
        if e.derivative_value is None:
            continue
        v = float(e.derivative_value)
        if not math.isfinite(v):
            continue
        av = abs(v)
        if av > best:
            best = av
    return best


def _max_distance(distance_map: dict[str, float]) -> float:
    """Return the largest finite distance over the map.

    Used by ``weight_mode="normalized"`` and ``"linear"`` for the
    distance-side normalisation.  Returns ``0.0`` when the map is empty
    or every distance is non-finite — the caller skips normalisation
    in that case.
    """
    best = 0.0
    for d in distance_map.values():
        if d is None:
            continue
        if not math.isfinite(d):
            continue
        if d > best:
            best = d
    return best


def _weighted_score(
    entry: CounterfactualEntry,
    distance_map: dict[str, float],
    *,
    weight_mode: str,
    max_grad: float,
    max_dist: float,
    inf_replacement: float,
) -> float:
    """Compute one entry's composite score under ``weight_mode``.

    All three modes return a non-negative scalar; higher score → more
    actionable (should rank earlier in the trace).  A missing distance
    or zero-magnitude gradient yields 0.0 — the row gets the lowest
    score and the actionable filter drops it before the cap.

    Mode semantics:

      * ``"distance"`` — pure multiplicative ``|∂s/∂p| · d_s``.  The
        natural composition: "this knob's leverage on the metric that
        is farthest from spec".  Used by default.
      * ``"normalized"`` — multiplicative ``(|∂s/∂p|/max_grad) · (d_s/max_dist)``.
        Both halves are scaled to ``[0, 1]`` before the product so an
        outlier in either dimension can't dominate.  Used when the
        cohort has heterogeneous knob kinds (W in µm scaling × Cc in
        pF scaling — raw magnitudes are not commensurable).
      * ``"linear"`` — additive ``0.5 · (|∂s/∂p|/max_grad) + 0.5 · (d_s/max_dist)``.
        Lossy but useful when you want any single dimension to lift
        a row (e.g. a knob with huge gradient on a satisfied spec
        still surfaces).  The two halves can be re-weighted by the
        caller via ``weights`` kwarg in the future; for v0.1.05 the
        split is fixed 50/50.
    """
    if entry.derivative_value is None:
        return 0.0
    g = float(entry.derivative_value)
    if not math.isfinite(g) or g == 0.0:
        return 0.0
    ag = abs(g)
    raw_distance = _lookup_distance(distance_map, entry.spec_metric)
    d = _normalize_distance_for_score(
        raw_distance, inf_replacement=inf_replacement,
    )
    if weight_mode == WEIGHT_MODE_DISTANCE:
        return ag * d
    if weight_mode == WEIGHT_MODE_NORMALIZED:
        # Avoid divide-by-zero: a maxes-of-zero map gives the entry a
        # bare gradient score so the row still surfaces.
        ag_n = ag / max_grad if max_grad > 0 else ag
        d_n = d / max_dist if max_dist > 0 else d
        return ag_n * d_n
    if weight_mode == WEIGHT_MODE_LINEAR:
        ag_n = ag / max_grad if max_grad > 0 else ag
        d_n = d / max_dist if max_dist > 0 else d
        return 0.5 * ag_n + 0.5 * d_n
    raise ValueError(
        f"unsupported weight_mode {weight_mode!r}.  Legal: {_WEIGHT_MODES}"
    )


def select_top_k_by_distance(
    entries: Iterable[CounterfactualEntry],
    violations: Iterable[Any],
    *,
    top_k: int = 8,
    abs_floor: float = 0.0,
    weight_mode: str = WEIGHT_MODE_DISTANCE,
    inf_replacement: float = 1e9,
) -> list[CounterfactualEntry]:
    """Filter and re-rank entries by ``|gradient| × distance_to_sat``.

    A weighted variant of :func:`select_top_k_actionable`.  Same
    actionable-row filter (``reason == "ok"``, finite non-zero
    derivative, ``abs(derivative_value) >= abs_floor``).  The
    difference is the sort key:

      * :func:`select_top_k_actionable` sorts by ``|derivative_value|``
        — "knobs with the most leverage on the metric".
      * :func:`select_top_k_by_distance` sorts by the composite
        ``|derivative_value| × distance_to_sat(spec_metric)`` — "knobs
        with the most leverage on the metric *that is farthest from
        spec*".  An LLM consuming this surfaces the knob that helps
        reduce the largest violation first.

    Parameters
    ----------
    entries
        Iterable of :class:`CounterfactualEntry`.  Order is irrelevant
        (the function sorts).
    violations
        Iterable of :class:`clay.distance_to_sat.SpecViolation`.  Only
        the ``spec_name`` and ``distance`` attributes are read; the
        function does NOT call into the simulator or the adjoint.
    top_k
        Maximum number of entries to return.  Default 8 (matches
        :func:`select_top_k_actionable`).
    abs_floor
        Drop entries with ``|derivative_value| < abs_floor``.  Default
        0.0 (no floor).  See :func:`select_top_k_actionable`.
    weight_mode
        One of ``"distance"`` / ``"normalized"`` / ``"linear"``.  See
        :func:`_weighted_score` for the per-mode formulas.  Default
        ``"distance"`` (raw multiplicative product — the philosophy
        supplement §5 spec-algebra default).
    inf_replacement
        Finite replacement for ``math.inf`` distance (the missing-
        measure case).  Default 1e9 — large enough to dwarf any
        finite distance, small enough that ``Σ score`` stays finite
        for downstream consumers.  Set to ``math.inf`` to let the
        infinite distance dominate naturally (Python sort handles it).

    Returns
    -------
    list[CounterfactualEntry]
        At most ``top_k`` entries, sorted by descending weighted score.
        Empty list when no actionable entries remain after filtering.

    Byte-exact note
    ---------------
    Does NOT modify :func:`select_top_k_actionable` or any other
    existing surface.  The R78 B prompt-formatter path is preserved
    unchanged — callers wanting the new ordering opt in explicitly.
    """
    if weight_mode not in _WEIGHT_MODES:
        raise ValueError(
            f"unsupported weight_mode {weight_mode!r}.  Legal: {_WEIGHT_MODES}"
        )
    # Filter to actionable rows (same predicate as select_top_k_actionable
    # so callers that switch modes mid-loop see identical row sets).
    filtered: list[CounterfactualEntry] = []
    for e in entries:
        if e.reason != "ok":
            continue
        if not e.is_actionable():
            continue
        if e.derivative_value is None:
            continue
        if abs(float(e.derivative_value)) < abs_floor:
            continue
        filtered.append(e)
    if not filtered:
        return []
    distance_map = _build_distance_map(violations)
    max_grad = _max_abs_gradient(filtered)
    max_dist = _max_distance(distance_map)
    # Compute score per entry, then sort.  Ties broken by:
    #   1. raw |gradient| descending  — keep R78 B ordering when distances tie
    #   2. spec_metric alphabetical    — deterministic across runs
    #   3. param_name alphabetical
    scored: list[tuple[float, float, str, str, CounterfactualEntry]] = []
    for e in filtered:
        score = _weighted_score(
            e, distance_map,
            weight_mode=weight_mode,
            max_grad=max_grad,
            max_dist=max_dist,
            inf_replacement=inf_replacement,
        )
        ag = abs(float(e.derivative_value or 0.0))
        scored.append((-score, -ag, e.spec_metric, e.param_name, e))
    scored.sort()
    out = [tup[4] for tup in scored]
    if top_k <= 0:
        return []
    return out[:top_k]


def compute_entry_score(
    entry: CounterfactualEntry,
    violations: Iterable[Any],
    *,
    weight_mode: str = WEIGHT_MODE_DISTANCE,
    inf_replacement: float = 1e9,
) -> float:
    """Return one entry's composite (gradient × distance) score.

    Public wrapper around :func:`_weighted_score` for callers that want
    the per-entry score without going through the top-K sort path.
    Useful for diagnostic / audit code paths that need to display the
    score in a UI alongside the ranked list.

    Returns the same ``0.0`` sentinel as the top-K path for non-
    actionable / missing-distance entries, so a caller can use a
    score-of-zero check as a "would-have-been-filtered" signal.
    """
    if weight_mode not in _WEIGHT_MODES:
        raise ValueError(
            f"unsupported weight_mode {weight_mode!r}.  Legal: {_WEIGHT_MODES}"
        )
    distance_map = _build_distance_map(violations)
    max_grad = _max_abs_gradient([entry])
    max_dist = _max_distance(distance_map)
    return _weighted_score(
        entry, distance_map,
        weight_mode=weight_mode,
        max_grad=max_grad,
        max_dist=max_dist,
        inf_replacement=inf_replacement,
    )


# ─── format_trace_for_llm_prompt ────────────────────────────────────


def _group_by_metric(
    entries: Iterable[CounterfactualEntry],
) -> list[tuple[str, list[CounterfactualEntry]]]:
    """Group entries by ``spec_metric`` preserving first-seen order."""
    seen: dict[str, list[CounterfactualEntry]] = {}
    order: list[str] = []
    for e in entries:
        if e.spec_metric not in seen:
            seen[e.spec_metric] = []
            order.append(e.spec_metric)
        seen[e.spec_metric].append(e)
    return [(m, seen[m]) for m in order]


# ─── v0.1.15 BL-124 R92 F: distance-weighted trace sort ──────────────


def sort_entries_by_distance_weight(
    entries: Iterable[CounterfactualEntry],
    violations: Iterable[Any],
    *,
    inf_replacement: float = 1e9,
) -> list[CounterfactualEntry]:
    """Re-order trace entries so the metric farthest from spec leads.

    BL-124: the LLM consuming a counterfactual trace should see the
    metric WORST-out-of-spec first, with its actionable knobs (highest
    ``|gradient|`` first) immediately below.  The R78 B sort surfaces
    knobs by ``|gradient|`` within each metric group but keeps the
    metric groups in first-seen order — which is arbitrary from the
    LLM's point of view.

    This helper is the BL-124 hook: it reads the same SpecViolation
    records the verifier already builds, computes a per-metric weight
    via :func:`clay.distance_to_sat.compute_metric_weight`, and returns
    a flat list of entries:

      * Metric groups sorted by ``max(distance)`` descending — the
        metric with the LARGEST distance leads.
      * Within each metric group, the R78 B ordering is preserved
        verbatim (``is_actionable`` first, then ``|derivative_value|``
        descending).  This keeps the per-metric KNOB ordering byte-exact
        with what :func:`compute_counterfactual_trace` already produced.
      * Ties on weight broken by:
          1. Metric having a `compute_failed` / `unsupported_metric`
             entry — these surface ABOVE satisfied metrics so the LLM
             sees the failure mode (the inf_replacement clamps these to
             the same weight as a missing measure).
          2. Alphabetical order of the metric name (deterministic
             across runs).

    Parameters
    ----------
    entries
        Iterable of :class:`CounterfactualEntry`.  Order is irrelevant
        — the function sorts.  Empty iterable returns ``[]``.
    violations
        Iterable of :class:`clay.distance_to_sat.SpecViolation`.  Empty
        iterable: every metric collapses to weight 0 and the function
        degenerates to the R78 B first-seen order (byte-exact).
    inf_replacement
        Finite replacement for ``math.inf`` distance.  Default ``1e9``
        — matches :func:`select_top_k_by_distance`.

    Returns
    -------
    list[CounterfactualEntry]
        Re-ordered list, same length as ``entries``.  Never drops a row
        — the BL-124 surface is ORDER-changing, not filtering.  Use
        :func:`select_top_k_by_distance` when you also want to truncate.

    Byte-exact note
    ---------------
    Does NOT mutate the input entries (the dataclass is frozen anyway).
    The legacy code paths that consume the un-sorted trace remain
    untouched — the BL-124 sort is an opt-in surfacing hook.
    """
    entries_list = list(entries)
    if not entries_list:
        return []
    # Build the (per-metric, max-distance) weight map.  Use the BL-124
    # helper so both this module and clay.distance_to_sat agree on the
    # alias-resolution rules in one place.
    from .distance_to_sat import compute_metric_weight

    grouped = _group_by_metric(entries_list)
    weights: dict[str, float] = {}
    for metric_name, _group in grouped:
        weights[metric_name] = compute_metric_weight(
            metric_name, list(violations), inf_replacement=inf_replacement,
        )
    # Tie-breaker: a group containing a non-actionable failure row
    # surfaces ABOVE a satisfied metric.  Encoded as a secondary key.
    def _has_failure(group: list[CounterfactualEntry]) -> bool:
        return any(e.reason != "ok" for e in group)

    # Sort key: (-weight, -has_failure_int, metric_name_lower).  Higher
    # weight first; failures lift a metric above satisfied peers; ties
    # broken alphabetically for determinism.
    sorted_groups = sorted(
        grouped,
        key=lambda mg: (
            -weights.get(mg[0], 0.0),
            -1 if _has_failure(mg[1]) else 0,
            mg[0].lower(),
        ),
    )
    flat: list[CounterfactualEntry] = []
    for _metric, group in sorted_groups:
        flat.extend(group)
    return flat


def trace_metric_weights(
    entries: Iterable[CounterfactualEntry],
    violations: Iterable[Any],
    *,
    inf_replacement: float = 1e9,
) -> dict[str, float]:
    """Return the per-metric distance weight used by the BL-124 sorter.

    Diagnostic surface mirroring :func:`compute_entry_score` — useful for
    audit logs and tests that want to introspect WHY a trace was
    re-ordered, not just SEE the result.

    Parameters
    ----------
    entries
        Source of metric names (the function only reads
        ``e.spec_metric``).
    violations
        SpecViolation records (read via
        :func:`clay.distance_to_sat.compute_metric_weight`).
    inf_replacement
        Forwarded.

    Returns
    -------
    dict[str, float]
        ``{metric_name: weight}`` — one entry per unique
        ``spec_metric`` seen in ``entries``.  The map preserves the
        original metric-name spelling from the entries (no alias
        rewrite) so a caller comparing to the entries themselves sees
        matching keys.
    """
    from .distance_to_sat import compute_metric_weight

    vio_list = list(violations)
    out: dict[str, float] = {}
    for e in entries:
        if e.spec_metric in out:
            continue
        out[e.spec_metric] = compute_metric_weight(
            e.spec_metric, vio_list, inf_replacement=inf_replacement,
        )
    return out


def _format_markdown_table(
    grouped: list[tuple[str, list[CounterfactualEntry]]],
) -> str:
    """Render the grouped entries as a single markdown table.

    Columns: knob | kind | ∂metric | value | unit | note

    The "note" column carries a "coerced from <kind>" tag when the
    bridge coerced an unknown kind to W, or the failure ``reason`` when
    the entry is non-actionable.  This keeps the LLM honest: it can see
    the gradient it asked for is missing (and why) instead of believing
    it always gets a number.
    """
    lines: list[str] = []
    header = (
        "| knob | kind | ∂metric | value | unit | note |"
    )
    sep = (
        "|------|------|---------|-------|------|------|"
    )
    lines.append(header)
    lines.append(sep)
    for _metric, group in grouped:
        for e in group:
            note_parts: list[str] = []
            if e.was_coerced:
                note_parts.append(f"coerced→{e.bridge_kind}")
            if e.reason != "ok":
                note_parts.append(e.reason)
            note = "; ".join(note_parts)
            value = e.render_value()
            unit = e.render_unit()
            lines.append(
                f"| {e.param_name} | {e.knob_kind} | "
                f"∂{e.spec_metric}/∂{e.knob_kind} | "
                f"{value} | {unit} | {note} |"
            )
    return "\n".join(lines)


def _format_plain_text(
    grouped: list[tuple[str, list[CounterfactualEntry]]],
) -> str:
    """Render the grouped entries as plain text bullets.

    Used by the "plain" style for prompts that are sensitive to
    markdown overhead (e.g. older free-form text models).  One bullet
    per row; bullets grouped under a per-metric header.
    """
    out: list[str] = []
    for metric, group in grouped:
        out.append(f"- spec metric: {metric}")
        for e in group:
            value = e.render_value() or "n/a"
            unit = e.render_unit()
            note_parts: list[str] = []
            if e.was_coerced:
                note_parts.append(f"coerced from {e.knob_kind} to {e.bridge_kind}")
            if e.reason != "ok":
                note_parts.append(e.reason)
            note = f" [{'; '.join(note_parts)}]" if note_parts else ""
            out.append(
                f"  * ∂{metric}/∂{e.knob_kind}({e.param_name}) = "
                f"{value} {unit}{note}"
            )
    return "\n".join(out)


def format_trace_for_llm_prompt(
    entries: Iterable[CounterfactualEntry],
    *,
    style: str = "markdown",
    include_legend: bool = True,
    violations: Optional[Iterable[Any]] = None,
    inf_replacement: float = 1e9,
) -> str:
    """Format a counterfactual trace as an LLM-readable string.

    Parameters
    ----------
    entries
        Iterable of :class:`CounterfactualEntry`.  Order is preserved
        within each metric group; metric groups appear in
        first-seen order (BL-124 default — see ``violations``).
    style
        ``"markdown"`` (default) — a single markdown table.
        ``"plain"`` — newline-separated bullets, no markdown.  Both
        styles include a one-line legend at the top when
        ``include_legend`` is True.
    include_legend
        Prepend a short legend explaining the sign convention and the
        rescaled units.  Default True.  Set False when the prompt
        template carries its own legend (avoids redundancy).
    violations
        v0.1.15 BL-124 opt-in surface.  When provided, the function
        re-orders metric groups by ``max(distance_to_sat)`` so the LLM
        sees the WORST-out-of-spec metric first; per-metric knob
        ordering is preserved byte-exact with the R78 B path.  When
        ``None`` (default), metrics appear in first-seen order — the
        R78 B byte-exact behavior is preserved.
    inf_replacement
        BL-124: finite replacement for ``math.inf`` distance when the
        re-order hook engages.  Default ``1e9``.  Ignored when
        ``violations is None``.

    Returns
    -------
    str
        Multi-line string suitable for embedding directly in an LLM
        prompt.  Empty trace renders an explicit "no counterfactual
        trace available" sentinel so the LLM doesn't see a blank
        section that looks like a truncation bug.

    Byte-exact note
    ---------------
    The legend wording, table column headers, separator row format,
    plain-text bullet shape — every byte of the rendered output is
    preserved when ``violations is None``.  Only the metric ORDER
    changes when ``violations`` is passed; the per-metric row content
    is identical to the R78 B path.
    """
    entries_list = list(entries)
    if not entries_list:
        return "(no counterfactual trace available)"
    # BL-124 hook: re-order entries so the metric farthest from spec
    # leads.  When violations is None the function degenerates to the
    # R78 B first-seen order — byte-exact preserved.
    if violations is not None:
        entries_list = sort_entries_by_distance_weight(
            entries_list, list(violations),
            inf_replacement=inf_replacement,
        )
    grouped = _group_by_metric(entries_list)
    legend = (
        "# Counterfactual trace\n"
        "# Each row shows ∂(spec_metric)/∂(knob) at the current "
        "operating point.\n"
        "# A positive value means: increasing the knob by one unit "
        "moves the metric in the listed direction.\n"
        "# Values are rescaled to engineering units (W → µm, Cc → pF, "
        "Vbias → µA).\n"
    ) if include_legend else ""
    if style == "plain":
        body = _format_plain_text(grouped)
    else:
        body = _format_markdown_table(grouped)
    return legend + body if legend else body


# ─── Lightweight serialisation helper ───────────────────────────────


def trace_to_json_dict(
    entries: Iterable[CounterfactualEntry],
) -> list[dict[str, Any]]:
    """Serialise a trace to a list of plain dicts.

    Used by the real_api_run usage_breakdown so the trace can travel
    through the persisted artifact.  Each dict has the same fields as
    :class:`CounterfactualEntry`.
    """
    out: list[dict[str, Any]] = []
    for e in entries:
        out.append(
            {
                "param_name": e.param_name,
                "knob_kind": e.knob_kind,
                "bridge_kind": e.bridge_kind,
                "derivative_value": (
                    float(e.derivative_value)
                    if e.derivative_value is not None
                    else None
                ),
                "units": e.units,
                "spec_metric": e.spec_metric,
                "was_coerced": bool(e.was_coerced),
                "reason": e.reason,
            }
        )
    return out
