"""
Compensate combinator — frequency compensation wrap (v0.0.89 + v0.1.12 split).

`compensate(stage, miller=Cc, zero=Rz)` or
`compensate(stage, target_pm_deg=PM)` wraps a stage with internal Miller
compensation. The lowering inserts a Miller cap (and optional nulling
resistor) between the inner stage's primary internal high-Z node and
its output port.

The auto-sizing path (Newton + bisection + Adjoint) lives in
`compensate_autosize.py` to keep this module under the < 1000 LOC
budget. This module imports the autosize entry-point lazily inside the
lowered builder to break the import cycle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..ir import Cell, Device, Spec
from ..stage_interface import ComposedStage
from ._common import StageRef, TemplateBuilder, _is_finite, _resolve


# ─── Compensate ─────────────────────────────────────────────────────

def compensate(
    stage_arg: StageRef,
    *,
    miller: Optional[float] = None,
    zero: Optional[float] = None,
    target_pm_deg: Optional[float] = None,
) -> ComposedStage:
    """`compensate(stage, miller=Cc, zero=Rz)` — frequency compensation wrap.

    Wraps a stage with internal Miller compensation. Two equivalent
    entry surfaces:

      * **Explicit sizing**:
          - `miller=Cc` — Miller cap value (Farads).
          - `zero=Rz`   — optional nulling resistor (Ohms) in series
                          with the Miller cap.
        At least one of `miller` or `zero` must be supplied.

      * **Auto-sizing** (v0.0.91):
          - `target_pm_deg=PM` — let the compiler derive Cc / Rz so
                                  the measured phase margin lands at
                                  `target_pm_deg ± 5%` after at most
                                  5 ngspice probes. Starts from
                                  `Cc=1 pF` / `Rz=10 kΩ` and bisects
                                  on Cc with Rz held proportional.
        Explicit `miller=` / `zero=` MUST NOT be supplied at the same
        time as `target_pm_deg=`. The auto-sizing path requires the
        wrapped stage to have a `build_ac_testbench` companion in
        `clay.templates.<template>`; if the testbench module is not
        importable, the search degrades to the initial seed values.

    The lowering stage adds a single cap (and optional res) between
    the stage's primary internal high-impedance node and its output.

    Type-check rules:
      * Exactly one of {explicit miller/zero, target_pm_deg} must be
        supplied (auto vs explicit is mutually exclusive).
      * `miller` / `zero` must be positive finite real numbers when
        supplied. `target_pm_deg` must be a positive finite real
        between 0 and 180.
    """
    s = _resolve(stage_arg)

    has_explicit = miller is not None or zero is not None
    has_auto = target_pm_deg is not None
    if not has_explicit and not has_auto:
        raise ValueError(
            "compensate() requires either explicit (miller=...,/zero=...) "
            "OR auto-sizing (target_pm_deg=...)"
        )
    if has_explicit and has_auto:
        raise ValueError(
            "compensate(): cannot combine explicit miller=/zero= with "
            "target_pm_deg=. Choose one entry surface."
        )

    if miller is not None:
        if not isinstance(miller, (int, float)) or float(miller) <= 0 \
                or not _is_finite(float(miller)):
            raise ValueError(
                f"compensate(miller=...) must be a positive finite "
                f"capacitance in Farads, got {miller!r}"
            )
    if zero is not None:
        if not isinstance(zero, (int, float)) or float(zero) <= 0 \
                or not _is_finite(float(zero)):
            raise ValueError(
                f"compensate(zero=...) must be a positive finite "
                f"resistance in Ohms, got {zero!r}"
            )
    if target_pm_deg is not None:
        if not isinstance(target_pm_deg, (int, float)) \
                or not _is_finite(float(target_pm_deg)) \
                or float(target_pm_deg) <= 0 \
                or float(target_pm_deg) >= 180:
            raise ValueError(
                f"compensate(target_pm_deg=...) must be a positive "
                f"finite real between 0 and 180 degrees, got "
                f"{target_pm_deg!r}"
            )

    meta: Dict[str, Any] = {}
    if miller is not None:
        meta["miller"] = float(miller)
    if zero is not None:
        meta["zero"] = float(zero)
    if target_pm_deg is not None:
        meta["target_pm_deg"] = float(target_pm_deg)
        # The auto-sizing run lands the derived numbers back on `meta`
        # under `derived_miller` / `derived_zero` so an inspector can
        # see what the search picked without re-running it. The actual
        # search happens at lowering time (lazily) — calling `compensate()`
        # is cheap; it's `lower_to_template(...)(...)` that triggers
        # ngspice probes.

    return ComposedStage(
        kind="compensate",
        interface=s.interface,
        parts=(s,),
        meta=meta,
    )


# ─── Compensate lowering (v0.0.89) ─────────────────────────────────


def _lower_compensate(
    inner: ComposedStage,
    meta: Dict[str, Any],
) -> TemplateBuilder:
    """Build the compensate builder.

    Inserts a Miller cap (and optional nulling resistor) from the
    inner stage's primary internal high-impedance node to its output
    port. The high-impedance node is detected heuristically by looking
    for the GATE net of the inner stage's final CS device — the
    canonical Mcs in every first-cohort 2-stage template. For 3-stage
    NMCF variants the gate of the *last* CS device (`Mcs3` / equivalent)
    is the right anchor.

    If the inner stage has NO recognisable CS device (e.g. a single-
    stage folded cascode), the Miller cap is added across the existing
    output and the first internal net found on the gate of any device
    whose drain is the output — same semantic, broader match.

    Args:
        inner: the wrapped ComposedStage.
        meta: carries `meta['miller']` (cap value, F) and/or
            `meta['zero']` (nulling resistor, Ω) for explicit sizing,
            or `meta['target_pm_deg']` for the v0.0.91 auto-sizing
            path. Auto-sizing populates `meta['derived_miller']` /
            `meta['derived_zero']` / `meta['auto_search_trace']`
            after the first build so an inspector can read the
            resolution without re-running the search.
    """
    from ._lower import lower_to_template
    inner_builder = lower_to_template(inner)
    target_pm = meta.get("target_pm_deg")

    def _builder(cell_name: str, spec: Spec, **kwargs: Any) -> Cell:
        # Auto-sizing path (v0.0.91): probe ngspice to find Cc / Rz
        # that land PM near `target_pm_deg`. Falls back to seed values
        # if probing fails (e.g. no testbench module importable, the
        # template is unknown to the probe machinery, ngspice not on
        # PATH, etc.).
        if target_pm is not None and "miller" not in meta:
            from .compensate_autosize import _autosize_compensate
            cc, rz, trace = _autosize_compensate(
                inner, cell_name, spec, kwargs, float(target_pm),
            )
            meta["derived_miller"] = cc
            meta["derived_zero"] = rz
            meta["auto_search_trace"] = trace
            miller_cc: Optional[float] = cc
            zero_rz: Optional[float] = rz
        else:
            miller_cc = meta.get("miller")
            zero_rz = meta.get("zero")

        inner_cell = inner_builder(cell_name, spec, **kwargs)

        if miller_cc is None and zero_rz is None:
            # `compensate(..., miller=None, zero=None)` is rejected at
            # call-site already; this branch is a defensive no-op.
            return inner_cell

        # Find the output port (single-ended sink for all first-cohort
        # stages). If the inner cell has no `out` port, skip silently.
        if "out" not in inner_cell.ports:
            return inner_cell

        # Locate the gate of the final CS device — the high-Z node we
        # connect the Miller cap to. Prefer an instance literally named
        # `Mcs`; for nested-Miller 3-stage variants the third gain
        # stage's gate is what we want. We reject any candidate whose
        # gate is a supply rail or a recognised bias rail (`pbias` /
        # `nbias`) because those are bias-set DC nets, not high-Z signal
        # nodes — a Miller cap to a bias rail is degenerate.
        bias_rail_names = {"vdd", "vss", "pbias", "nbias"}
        hi_z_node: Optional[str] = None
        for d in inner_cell.devices:
            if d.instance_name == "Mcs":
                candidate = d.pins.get("g")
                if candidate is not None and candidate not in bias_rail_names:
                    hi_z_node = candidate
                    break
        if hi_z_node is None:
            # Fallback: find any nmos/pmos device whose drain is `out`
            # and whose gate is NOT a bias rail — that's a real gain
            # device whose gate is a high-Z internal node.
            for d in inner_cell.devices:
                if d.device_class not in ("nmos", "pmos"):
                    continue
                if d.pins.get("d") != "out":
                    continue
                candidate = d.pins.get("g")
                if candidate is None or candidate in bias_rail_names:
                    continue
                # Also reject `in` / `inn` / `inp` (cell-external input
                # ports) — Miller cap from out to an INPUT port is a
                # full-loop feedback, not a Miller pole-splitter.
                if candidate in inner_cell.ports:
                    continue
                hi_z_node = candidate
                break

        if hi_z_node is None:
            # No sensible high-Z node — degrade to a no-op rather than
            # adding a meaningless device. Single-stage topologies
            # (e.g. single CS amp) fall through here.
            return inner_cell

        new_devices = list(inner_cell.devices)

        # Compute the cap / resistor net topology. If both miller and
        # zero are present: out → Cmiller → nz → Rz → hi_z_node. If only
        # miller: out → Cmiller → hi_z_node. If only zero: out → Rz →
        # hi_z_node (no cap — uncommon but well-defined).
        if miller_cc is not None and zero_rz is not None:
            nz_node = "comp_nullz"
            new_devices.append(Device(
                instance_name="Cmiller",
                device_class="cap",
                pdk_variant=None,
                pins={"p": "out", "n": nz_node},
                sizing={"value": float(miller_cc)},
            ))
            new_devices.append(Device(
                instance_name="Rnullz",
                device_class="res",
                pdk_variant=None,
                pins={"p": nz_node, "n": hi_z_node},
                sizing={"value": float(zero_rz)},
            ))
        elif miller_cc is not None:
            new_devices.append(Device(
                instance_name="Cmiller",
                device_class="cap",
                pdk_variant=None,
                pins={"p": "out", "n": hi_z_node},
                sizing={"value": float(miller_cc)},
            ))
        else:
            new_devices.append(Device(
                instance_name="Rnullz",
                device_class="res",
                pdk_variant=None,
                pins={"p": "out", "n": hi_z_node},
                sizing={"value": float(zero_rz)},
            ))

        return Cell(
            name=inner_cell.name,
            ports=list(inner_cell.ports),
            spec=spec,
            devices=new_devices,
            pdk_id=inner_cell.pdk_id,
        )

    _builder.__name__ = f"compensate__{inner.template_name}"
    return _builder
