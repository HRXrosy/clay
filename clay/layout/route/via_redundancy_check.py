"""VIA-REDUNDANCY obligation -- an INDEPENDENT checker read from emitted GDS.

THE OBLIGATION
==============
The reference layout stitches every inter-metal connection with a redundant via
ARRAY, not a single cut. Measured on the reference (``reference_layout.gds``):
median 4 cuts/connection on via1, 5 on via2, minimum 2 on both. This is the
standard reliability practice -- electromigration headroom, contact-resistance
averaging, and open-defect tolerance (one voided cut in an array of four still
conducts). A router that drops a SINGLE cut per layer transition (measured
median 1 on our generic route) leaves every connection one void away from an
open. This checker names that as a checkable obligation: each inter-metal
connection must carry AT LEAST the reference-calibrated redundancy.

RELATIVE, REFERENCE-CALIBRATED (no absolute magic gate)
=======================================================
The pass/fail band is a PERCENTAGE tolerance of a TARGET the REFERENCE layout
calibrates -- measured live from a supplied ``reference_gds_path`` (the preferred
path, a genuine relative comparison) or, when none is supplied, the module's
calibrated default (``DEFAULT_TARGET_VIAS_PER_CONN`` -- provenance: the reference
layout's measured MINIMUM redundancy, 2 cuts/connection). Either way the verdict
is ``measured_median >= target * (1 - rel_tol)`` -- a RELATIVE band, never a
hardcoded absolute cut count as the sole gate. The reference calibrates the
checker; it NEVER feeds the engine.

INDEPENDENCE (checker not the checked)
======================================
Reads the EMITTED GDS only -- never the router's internal state. For each via
layer the PDK declares it clusters the cut polygons into CONNECTIONS (an array's
cuts sit within one via-space rule of each other; distinct connections are
microns apart) with the KLayout grow+merge idiom, then takes the MEDIAN cuts per
connection -- the SAME metric the reference reports. It consults the PDK ``tech``
only for the via layer keys + the ``via{k}_space`` rule value (the SAME rule the
array materializer uses to set the cut pitch). A divergence between this checker
and a real reliability sign-off is a bug in this checker, never grounds to trust
it over the oracle.

GENERIC, NO PER-PDK BRANCH
==========================
Via layers + the via-space rule come entirely from the PDK contract
(``tech.via_key`` / ``tech.via_size`` / ``tech.drc_rules`` / ``tech.via_rule``);
a PDK with a different stack or spacing is checked by the same code with its own
numbers -- zero ``if pdk == ...`` branch.

FIVE-STATE (never silent-pass)
==============================
``pass`` / ``fail`` / ``unmeasurable`` (no klayout, unreadable GDS, or no via
cuts present at all) / ``unsupported`` (the PDK declares no via layers, so the
obligation does not apply) / ``advisory`` (cuts measured but no reference and no
target to judge against -- reported, never passed).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LayerKey = Tuple[int, int]

# Calibrated default target when no reference GDS is supplied at gate time.
# PROVENANCE (measured, not invented): the reference layout
# ``ota_dev/ref/reference_layout.gds`` carries a MINIMUM of 2 cuts per connection
# on both via1 and via2 (median 4 / 5). 2.0 is the reference's floor redundancy,
# NOT an engine input. The gate is still RELATIVE (a percentage band of this
# target), so this is a calibration constant, never the sole absolute-count gate.
DEFAULT_TARGET_VIAS_PER_CONN = 2.0

# Percentage band around the target (relative-thresholds-only house rule). A
# single-via layout (median 1) falls below 2.0*(1-0.25)=1.5 -> RED; any genuine
# >=2 redundancy clears it -> GREEN.
DEFAULT_REL_TOL = 0.25


@dataclass
class ViaLayerRedundancy:
    """Per-via-layer redundancy measurement + verdict.

    ``vias_per_connection`` is the MEDIAN cut count across the layer's connection
    clusters (the reference-comparable metric); ``target`` / ``floor`` are the
    calibrated target and its relative-band floor; ``ok`` is
    ``vias_per_connection >= floor`` (``None`` when no target resolved -> the
    layer is advisory, not judged)."""

    via_name: str
    via_layer: _LayerKey
    n_cuts: int
    n_connections: int
    vias_per_connection: float
    mean: float
    min_per_conn: int
    max_per_conn: int
    target: Optional[float] = None
    floor: Optional[float] = None
    ok: Optional[bool] = None


@dataclass
class ViaRedundancyReport:
    """Result of :func:`check_via_redundancy` on one GDS.

    ``status`` is one of ``pass`` / ``fail`` / ``unmeasurable`` / ``unsupported``
    / ``advisory`` (never a silent pass). ``per_layer`` carries each measured via
    layer; ``target_source`` records how the target was calibrated
    (``"reference-gds"`` / ``"calibrated-default"`` / ``"none"``)."""

    status: str = "unmeasurable"
    per_layer: List[ViaLayerRedundancy] = field(default_factory=list)
    rel_tol: float = DEFAULT_REL_TOL
    target_source: str = "none"
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def failing_layers(self) -> List[ViaLayerRedundancy]:
        return [l for l in self.per_layer if l.ok is False]


def _via_layers(tech: Any, max_via: int) -> List[Tuple[str, _LayerKey]]:
    """The ``(via_name, gds_key)`` of each inter-metal cut layer the PDK declares,
    generically (``via1`` .. ``via{max_via}``). Empty -> the obligation is
    unsupported for this PDK."""
    out: List[Tuple[str, _LayerKey]] = []
    for n in range(1, max_via + 1):
        try:
            key = tuple(tech.via_key(n))  # type: ignore[arg-type]
        except Exception:
            continue
        if key is None:
            continue
        out.append((f"via{n}", key))  # type: ignore[arg-type]
    return out


def _via_space_nm(tech: Any, via_name: str) -> int:
    """The ``via{k}_space`` rule value (nm) the PDK declares, used as the cluster
    grow distance (adjacent array cuts sit exactly this far apart). Reads the SAME
    rule source the materializer uses: parsed ``drc_rules`` first, then
    ``via_rule`` spacing, then the cut size, then a conservative floor. No
    constants of its own beyond the last-resort floor."""
    try:
        parsed = tech.drc_rules() or {}
    except Exception:
        parsed = {}
    e = parsed.get(f"{via_name}_space")
    if isinstance(e, dict):
        v = e.get("value_nm")
        if v is None and e.get("value_um") is not None:
            v = round(float(e["value_um"]) * 1000.0)
        if v:
            return int(v)
    try:
        vr = tech.via_rule(via_name)
    except Exception:
        vr = None
    sp = getattr(vr, "spacing", None) if vr is not None else None
    if sp is not None:
        v = getattr(sp, "value", sp)
        if v:
            return int(v)
    try:
        cut = tech.via_size(via_name)
    except Exception:
        cut = None
    if cut:
        return int(cut)
    return 200  # last-resort clustering floor (a via cut is never smaller)


def _measure_layer(cell, ly, key: _LayerKey, grow_nm: int):
    """Cluster the cut polygons on ``key`` into connections and return
    ``(n_cuts, n_connections, per_connection_counts)`` or ``None`` when the layer
    is absent / empty. A cut is grown by ``grow_nm`` and merged: array cuts (edge
    gap == via-space) chain into one connection; distinct connections (microns
    apart) stay separate. Each connection's cut count is the number of original
    cuts interacting with the merged cluster."""
    import klayout.db as kdb

    li = ly.find_layer(int(key[0]), int(key[1]))
    if li is None:
        return None
    reg = kdb.Region(cell.begin_shapes_rec(li))
    n_cuts = reg.count()
    if n_cuts == 0:
        return None
    grown = reg.dup()
    grown.size(int(grow_nm))
    grown.merge()
    per: List[int] = []
    for cluster in grown.each():
        per.append(reg.interacting(kdb.Region(cluster)).count())
    return n_cuts, len(per), per


def _measure_all(gds_path: str, tech: Any, top_cell: Optional[str], max_via: int):
    """Measure ``{via_name: (key, n_cuts, per_connection_counts)}`` on ``gds_path``.

    Returns ``(measurements, via_layers, reason)``. ``measurements`` is empty when
    no via cuts are present; ``via_layers`` is the PDK's declared via-layer set
    (empty -> unsupported). ``reason`` is a non-empty string on a hard read /
    klayout failure (-> unmeasurable)."""
    try:
        import klayout.db as kdb
    except Exception as exc:  # pragma: no cover - klayout present in CI
        return None, [], f"klayout import: {exc}"

    via_layers = _via_layers(tech, max_via)
    if not via_layers:
        return {}, [], "pdk declares no via layers"

    ly = kdb.Layout()
    try:
        ly.read(gds_path)
    except Exception as exc:
        return None, via_layers, f"read {gds_path}: {exc}"

    if top_cell is not None:
        cell = ly.cell(top_cell)
        if cell is None:
            return None, via_layers, f"top cell {top_cell!r} not found"
    else:
        tops = ly.top_cells()
        if not tops:
            return None, via_layers, "no top cell"
        cell = tops[0]

    out: Dict[str, Tuple[_LayerKey, int, List[int]]] = {}
    for via_name, key in via_layers:
        grow = _via_space_nm(tech, via_name)
        m = _measure_layer(cell, ly, key, grow)
        if m is None:
            continue
        n_cuts, _n_conn, per = m
        out[via_name] = (key, n_cuts, per)
    return out, via_layers, ""


def check_via_redundancy(
    gds_path: str,
    tech: Any,
    *,
    reference_gds_path: Optional[str] = None,
    target_vias_per_conn: Optional[float] = DEFAULT_TARGET_VIAS_PER_CONN,
    rel_tol: float = DEFAULT_REL_TOL,
    top_cell: Optional[str] = None,
    reference_top_cell: Optional[str] = None,
    max_via: int = 6,
) -> ViaRedundancyReport:
    """Measure the via-redundancy obligation on ``gds_path`` against ``tech``.

    For each via layer the PDK declares, cluster the emitted cut polygons into
    connections and take the MEDIAN cuts per connection. Resolve a TARGET
    redundancy:

      * ``reference_gds_path`` supplied -> measure that reference's per-layer
        median as the target (the preferred, genuinely RELATIVE calibration);
      * else ``target_vias_per_conn`` (default the reference-calibrated
        ``DEFAULT_TARGET_VIAS_PER_CONN``) -> a uniform target;
      * else (explicit ``None``, no reference) -> ``advisory`` (measured, not
        judged -- never a silent pass).

    A via layer passes when ``median >= target * (1 - rel_tol)``. Overall
    ``status`` is ``fail`` if any measured layer falls below its floor, ``pass``
    if all measured layers clear it, ``unmeasurable`` when no via cuts / klayout
    are present, ``unsupported`` when the PDK declares no via layers.
    """
    report = ViaRedundancyReport(rel_tol=float(rel_tol))

    meas, via_layers, reason = _measure_all(gds_path, tech, top_cell, max_via)
    if not via_layers:
        report.status = "unsupported"
        report.reason = reason or "pdk declares no via layers"
        return report
    if meas is None:
        report.status = "unmeasurable"
        report.reason = reason or "gds unreadable"
        return report
    if not meas:
        report.status = "unmeasurable"
        report.reason = "no via cuts present in GDS"
        return report

    # --- resolve the per-layer target (reference calibrates, never the engine).
    per_layer_target: Dict[str, float] = {}
    target_source = "none"
    if reference_gds_path:
        ref_meas, _rvl, _rr = _measure_all(
            reference_gds_path, tech, reference_top_cell, max_via
        )
        if ref_meas:
            for via_name, (_k, _n, ref_per) in ref_meas.items():
                if ref_per:
                    per_layer_target[via_name] = float(statistics.median(ref_per))
            if per_layer_target:
                target_source = "reference-gds"
    if target_source == "none" and target_vias_per_conn is not None:
        for via_name in meas:
            per_layer_target[via_name] = float(target_vias_per_conn)
        target_source = "calibrated-default"
    report.target_source = target_source

    any_judged = False
    any_fail = False
    for via_name, (key, n_cuts, per) in sorted(meas.items()):
        median = float(statistics.median(per)) if per else 0.0
        layer = ViaLayerRedundancy(
            via_name=via_name,
            via_layer=key,
            n_cuts=n_cuts,
            n_connections=len(per),
            vias_per_connection=median,
            mean=(float(statistics.mean(per)) if per else 0.0),
            min_per_conn=(min(per) if per else 0),
            max_per_conn=(max(per) if per else 0),
        )
        target = per_layer_target.get(via_name)
        if target is not None:
            floor = target * (1.0 - float(rel_tol))
            layer.target = target
            layer.floor = floor
            layer.ok = median >= floor
            any_judged = True
            if layer.ok is False:
                any_fail = True
        report.per_layer.append(layer)

    if not any_judged:
        report.status = "advisory"
        report.reason = "no reference or target to judge redundancy against"
        return report
    report.status = "fail" if any_fail else "pass"
    return report


__all__ = [
    "DEFAULT_TARGET_VIAS_PER_CONN",
    "DEFAULT_REL_TOL",
    "ViaLayerRedundancy",
    "ViaRedundancyReport",
    "check_via_redundancy",
]
