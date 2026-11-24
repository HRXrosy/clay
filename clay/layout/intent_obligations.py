"""Layout-intent OBLIGATION CHECKER: turn declared budgets into PEX verdicts.

The :class:`~clay.layout.intent.LayoutIntent` declares RELATIVE-imbalance budgets
(matched-cap balance per diff pair, input kickback, clk->regen coupling). This
module measures each budget against a real :class:`~clay.layout.pex.base.PexReport`
and emits one :class:`~clay.layout.intent.LayoutObligation` per budget, so the
intent contract becomes a pass/fail verification over the extracted parasitics.

Every measurement delegates to ``clay.layout.comparator_metrics`` so the
denominator conventions stay shared with the metrics layer:

  * ``matched_cc_balance`` -> ``differential_balance`` (per-pair total-cap
    imbalance, ``|Ca-Cb| / max(Ca,Cb)``);
  * ``kickback``           -> ``kickback_cc(...)["imbalance"]`` (summed-aggressor
    coupling onto each input, ``|c0-c1| / mean``);
  * ``clk_regen``          -> ``clk_regen_balance(...)["imbalance"]`` (clk coupling
    into crossn vs crossp, ``|cn-cp| / mean``).

Because everything reads a :class:`PexReport`, the checker is unit-testable
WITHOUT Magic/Docker by passing a synthetic report. All thresholds are RELATIVE
fractions (project-wide rule: never absolute aF deltas), so scaling every cap by
a constant leaves every verdict unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from .comparator_metrics import (
    DYNAMIC_AGGRESSORS as _LEGACY_DYNAMIC_AGGRESSORS,
    clk_regen_balance,
    differential_balance,
    kickback_cc,
    shielding_between,
)
from .intent import (
    LayoutIntent,
    LayoutObligation,
    LdeBudget,
    ParasiticBudget,
    place_group_floorplan_row,
    parasitic_cap_is_single_ended,
    _is_rail,
)
from .pex.base import PexCap, PexReport

# Default RELATIVE tolerances for the STRUCTURAL geometric obligations (fractions
# in (0, 1); relative-only, per the project-wide rule). A correctly built common-
# centroid cluster / matched pair holds its sub-centroids to WELL within these.
_DEFAULT_INTERDIG_TOL = 0.05
_DEFAULT_CC_TOL = 0.05
_DEFAULT_PLACE_GROUP_TOL = 0.10
# Declared @equal_size group: every member shares an IDENTICAL placed footprint
# (same width AND height) within a small RELATIVE spread (project-wide relative-only
# rule). A correctly realised equal-footprint group holds well within this; an
# unequal placement blows past it.
_DEFAULT_EQUAL_SIZE_TOL = 0.02
# Declared wire-width: exact-or-greater with a small RELATIVE under-width slack so a
# one-grid rounding of the centred wire rectangle never trips a physically-met width.
_DEFAULT_WIRE_WIDTH_TOL = 0.02
# Declared matched wire-LENGTH: two matched / differential nets' routed run lengths
# must be balanced within a RELATIVE imbalance slack (10%) so their wire parasitic
# (R / C) is balanced. This checker does not route; the same canonical pair is
# independently consumed as a hard Length constraint by the router.
_DEFAULT_MATCH_LENGTH_TOL = 0.10
# Declared @order(a, b, rel): the realised centroids must respect the relative order
# (left_of / above). The metric is the RELATIVE violation distance; an order is
# satisfied iff the violation is zero (a tie is not a violation), so the tol is 0.
_DEFAULT_ORDER_TOL = 0.0
# Declared @abut(a, b): the two devices' facing-edge gap must be at (near) the
# DRC-minimum -- a small RELATIVE slack on the larger device's facing extent.
_DEFAULT_ABUT_TOL = 0.15
# Declared @fix(device, x, y): the realised centroid must sit within one grid quantum
# of the pin. ``check_fix`` returns the centroid-to-pin offset DIVIDED by its absolute
# grid tolerance, so the pass threshold here is 1.0 (offset within tolerance). This is
# the only ABSOLUTE obligation (an absolute pin is a coordinate claim, not a relation).
_DEFAULT_FIX_TOL = 1.0
# Microns -> nm, to scale the @fix pin (authored in microns) onto the placement nm grid.
_FIX_UM_TO_NM = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Device-environment (LDE) geometry threading
# ─────────────────────────────────────────────────────────────────────────────
#
# An ``LdeBudget`` is measured over DEVICE GEOMETRY (per-device terminal bboxes +
# the cell's layer polygons), NOT over the PexReport caps. ``check_layout_intent``
# therefore accepts an optional :class:`LdeGeometry` carrying exactly what
# ``clay.layout.device_environment.extract_device_environment`` needs. When no
# geometry is supplied, every LDE budget is UNMEASURABLE (token withheld) -- the
# same blocking-on-missing-substrate policy a missing net gets, never a free pass.
# The cap-budget path is untouched by this parameter.


@dataclass(frozen=True)
class LdeGeometry:
    """The device geometry an :class:`LdeBudget` measures against.

    Mirrors the :func:`clay.layout.device_environment.extract_device_environment`
    signature so the obligation checker can hand it straight through:

      * ``devices``       -- ``{device_label: terminal-bbox-dict}`` (the router's
        ``placement.term`` shape) OR ``{label: active-bbox 4-tuple}``;
      * ``layer_polygons``-- the cell's flat ``LayerPolygon`` list;
      * ``pdk_layers``    -- the PDK ``LayerMap`` (PDK-neutral layer selection);
      * ``pmos_labels``   -- optional set of PMOS device labels (well-edge is
        measured against NWELL for these, PWELL otherwise).

    A factory :meth:`from_cell` reads the same ``placement.term`` /
    ``layer_polygons`` a routed cell exposes, so the certify path can build it
    from a real cell exactly the way ``placement_symmetry`` reads geometry.
    """

    devices: Dict[str, object]
    layer_polygons: Sequence[object]
    pdk_layers: object
    pmos_labels: Optional[Tuple[str, ...]] = None

    @staticmethod
    def from_cell(routed_cell, pdk_layers, *, pmos_labels=None) -> "LdeGeometry":
        """Build an :class:`LdeGeometry` from a routed cell + its PDK LayerMap.

        Reads the per-device terminal bboxes off ``cell.placement.term`` (falling
        back to ``cell.term``) and the cell's polygons off ``cell.layer_polygons``
        (falling back to ``cell.polygons``) -- the same geometry surfaces
        ``placement_symmetry`` and the GDS lowering already use.
        """
        placement = getattr(routed_cell, "placement", None)
        term = getattr(placement, "term", None)
        if term is None:
            term = getattr(routed_cell, "term", None)
        if term is None:
            raise TypeError(
                "routed_cell exposes neither .placement.term nor .term; cannot "
                "read device placements for LDE measurement"
            )
        # Layer geometry (well/diff polygons) is NOT carried on the routed cell --
        # it comes from lowering the placement to GDS shapes (backend.lower_*). When
        # it is not threaded in, do NOT crash: build geometry with no polygons so the
        # extractor honestly yields None for every well/diff metric -> the obligation
        # is UNMEASURABLE (token withheld), never a fabricated pass. A caller that has
        # already lowered the cell passes the polygons (.layer_polygons / .polygons)
        # in the SAME coordinate frame as placement.term.
        # NOTE: threading the lowered, frame-consistent polygons from the flow into
        # the certify path is the remaining integration; until then a real cell with
        # no polygons attached resolves UNMEASURABLE (honest) rather than measuring
        # against wrong-frame geometry.
        polys = getattr(routed_cell, "layer_polygons", None)
        if polys is None:
            polys = getattr(routed_cell, "polygons", None)
        if polys is None:
            polys = ()
        # PMOS roles: prefer an explicit list, else the cell's OWN device kinds (the
        # real classification) before the extractor falls back to a geometry-only
        # nwell-overlap heuristic.
        if pmos_labels is None:
            kinds = getattr(routed_cell, "device_kinds", None) or getattr(
                placement, "device_kinds", None
            )
            if isinstance(kinds, dict):
                pmos_labels = tuple(
                    l for l, k in kinds.items() if str(k).lower() == "pmos"
                )
        return LdeGeometry(
            devices=dict(term),
            layer_polygons=list(polys),
            pdk_layers=pdk_layers,
            pmos_labels=None if pmos_labels is None else tuple(pmos_labels),
        )


# Trailing hierarchical decoration Magic appends to a base net name when it
# flattens / renames extracted nets: `vout.n0`, `ntail.t1`, `vbias_p.n3`, and the
# bare-numeric `vmid_left1.0` form. A declared INTERNAL net `X` will not appear
# verbatim in such an extraction, so the checker resolves `X` to the PEX net(s)
# whose BASE name (this suffix stripped) equals `X`, case-insensitively.
_HIER_SUFFIX_RE = re.compile(r"\.(?:[nt]\d+|\d+)$", re.IGNORECASE)


def _base_net(name: str) -> str:
    """Strip a trailing `.n<digits>` / `.t<digits>` / `.<digits>` decoration.

    Magic decorates a hierarchically-resolved net with one such suffix; the base
    name (everything before it) is what a declared `.clay` net actually names.
    Only the FINAL suffix is stripped (a single level of decoration), so a net
    that genuinely contains a dot in its base (e.g. `x1.vout`) keeps `x1.` and
    only loses the trailing `.n0`.
    """
    return _HIER_SUFFIX_RE.sub("", str(name))


def build_net_resolution(report: PexReport) -> Dict[str, Tuple[str, ...]]:
    """Map a base net name -> the PEX nets that resolve to it.

    For every net actually present in ``report``, index it under both its
    verbatim name and its base name (suffix-stripped), case-insensitively. A
    declared net is resolved by looking up its lower-cased name in this map: an
    exact verbatim hit wins; otherwise every PEX net sharing the declared net's
    base name is returned (aggregated by the caller). Returns lower-cased keys ->
    tuple of the ORIGINAL-case PEX net names (deterministic, first-seen order).
    """
    present = sorted(report.nets())
    by_exact: Dict[str, List[str]] = {}
    by_base: Dict[str, List[str]] = {}
    for net in present:
        by_exact.setdefault(net.lower(), []).append(net)
        by_base.setdefault(_base_net(net).lower(), []).append(net)
    out: Dict[str, Tuple[str, ...]] = {}
    # Exact (verbatim) keys take precedence; base keys fill in the rest.
    for key, nets in by_base.items():
        out[key] = tuple(nets)
    for key, nets in by_exact.items():
        out[key] = tuple(nets)
    return out


def resolve_distributed_net_members(
    report: PexReport, declared: str
) -> Tuple[str, ...]:
    """Return every extracted node belonging to one declared distributed net.

    Unlike :func:`build_net_resolution`, this deliberately does not let an exact
    root-node hit hide its ``.n#``/``.t#`` children: distributed resistance and
    whole-net capacitance need the complete conductor, including its root.
    """

    key = str(declared).casefold()
    return tuple(
        sorted(
            net
            for net in report.nets()
            if net.casefold() == key or _base_net(net).casefold() == key
        )
    )


def _resolve_net(
    declared: str, resolution: Dict[str, Tuple[str, ...]]
) -> Tuple[str, ...]:
    """Resolve one declared net to the PEX nets it maps to (empty if none)."""
    return resolution.get(str(declared).lower(), ())


def resolve_report_for_intent(
    report: PexReport,
    intent: Optional[LayoutIntent],
    budgets,
) -> Tuple[PexReport, Dict[str, Tuple[str, ...]]]:
    """Return a report whose nets are RE-KEYED to the declared base names.

    Resolve each declared conductor with ``resolve_distributed_net_members``:
    an exact root and its extracted segments all contribute. The measurement
    view retains signed external capacitance and inter-net coupling, while
    omitting capacitance whose two terminals belong to the same conductor.
    The original extraction is not changed.

    Returns ``(resolved_report, resolution_map)``. Declared names map to their
    full extracted membership. A name with conflicting ownership or a resistor
    crossing its membership boundary maps to an empty tuple; the checker must
    withhold its measurement. Missing names have no entry. Cap-only extraction
    still uses the established suffix resolution without requiring resistors.
    """
    resolution = build_net_resolution(report)

    # Collect every declared name the obligations may read.
    declared: List[str] = []
    for b in budgets:
        declared.extend(b.pair)
        declared.extend(_required_nets(b, intent))
    if intent is not None:
        declared.extend(intent.dynamic_nets)

    members_by_name = {
        name: frozenset(resolve_distributed_net_members(report, name))
        for name in dict.fromkeys(declared)
    }
    owners: Dict[str, set[str]] = {}
    invalid = set()
    for name, members in members_by_name.items():
        for pex_net in members:
            owners.setdefault(pex_net, set()).add(name.lower())
        # Same boundary as electrical_budget._net_measurements: do not infer
        # membership by following a resistor into another named conductor.
        if any((r.a in members) != (r.b in members) for r in report.resistors):
            invalid.add(name.lower())
    for names in owners.values():
        if len(names) > 1:
            invalid.update(names)

    rename: Dict[str, str] = {}
    for name, members in members_by_name.items():
        key = name.lower()
        if key in invalid:
            resolution[key] = ()
        elif members:
            resolution[key] = tuple(sorted(members))
            for pex_net in members:
                rename[pex_net] = name
    if not rename:
        return report, resolution

    new_caps = tuple(
        PexCap(rename.get(c.a, c.a), rename.get(c.b, c.b), c.cap_aF)
        for c in report.caps
        if not (
            c.a in rename and c.b in rename and rename[c.a] == rename[c.b]
        )
    )
    new_res = tuple(
        replace(r, a=rename.get(r.a, r.a), b=rename.get(r.b, r.b))
        for r in report.resistors
    )
    resolved = replace(report, caps=new_caps, resistors=new_res)
    return resolved, resolution


def _budget_aggressors(
    budget: ParasiticBudget,
    intent: Optional[LayoutIntent],
) -> tuple:
    """Resolve the aggressor list for a kickback measurement.

    Resolution order:

      1. ``budget.aggressors`` if explicitly set (the budget owns its drivers);
      2. ``intent.dynamic_nets`` if the intent classifies any (the auto-derived
         policy: every dynamic net is an aggressor unless the budget says
         otherwise);
      3. the legacy comparator-specific ``DYNAMIC_AGGRESSORS`` constant -- ONLY
         the back-compat fallback for a caller that hand-built a LayoutIntent
         with empty ``dynamic_nets`` (e.g. an old test).

    Returns the resolved tuple. An EXPLICIT empty tuple on the budget is honoured
    (the caller said "no aggressors", which is a vacuously-balanced kickback,
    not a default).
    """
    if budget.aggressors is not None:
        return tuple(budget.aggressors)
    if intent is not None and intent.dynamic_nets:
        return tuple(intent.dynamic_nets)
    return tuple(_LEGACY_DYNAMIC_AGGRESSORS)


def _budget_clock_net(budget: ParasiticBudget) -> str:
    """Resolve the clock net for a clk_regen measurement.

    Honours ``budget.clock_net`` (the v2 driver); falls back to ``"clk"`` for the
    legacy comparator convention so existing call sites stay byte-identical.
    """
    return budget.clock_net or "clk"


def _single_ended_signal_and_rail(pair: Tuple[str, str]) -> Tuple[str, str]:
    """Split a single-ended ``parasitic_cap`` pair into (signal_net, rail_net).

    Exactly one of the two nets is a supply/ground rail (the caller guarantees
    single-endedness). When BOTH are rails the first is treated as the rail
    reference and the second as the (degenerate) signal node -- a harmless
    convention since such a budget is vacuous in practice.
    """
    a, b = pair
    if _is_rail(b):
        return a, b
    return b, a


def _measure_single_ended_parasitic_cap(
    report: PexReport,
    pair: Tuple[str, str],
) -> float:
    """RELATIVE off-rail coupling fraction for a single-ended parasitic_cap budget.

    A ``parasitic_cap(net, rail)`` budget is single-ended: it constrains how well
    the signal ``net`` is referenced to its intended supply/ground ``rail``, NOT a
    symmetry between two signal nodes (which the differential pair-imbalance metric
    measures, and which would report a meaningless ~1.0 false-fail for a
    signal-vs-rail pair). The measured quantity is the fraction of ``net``'s TOTAL
    parasitic capacitance that does NOT terminate on ``rail``::

        off_rail = (C_total(net) - C(net, rail)) / C_total(net)

    This is RELATIVE (a fraction in [0, 1]) and scale-invariant -- scaling every
    cap by a constant leaves it unchanged -- so it obeys the project-wide
    relative-only rule. ``0`` means every bit of ``net``'s parasitic references the
    rail (ideal); ``1`` means none of it does. A net with no extracted cap at all
    reports ``0`` (vacuously referenced; the unmeasurable gate upstream handles a
    net that is entirely absent).
    """
    signal, rail = _single_ended_signal_and_rail(pair)
    total = report.net_total_cap_aF(signal)
    if total <= 0.0:
        return 0.0
    to_rail = sum(
        c.cap_aF
        for c in report.caps
        if frozenset((c.a, c.b)) == frozenset((signal, rail))
    )
    return max(0.0, (total - to_rail) / total)


def _required_nets(
    budget: ParasiticBudget,
    intent: Optional[LayoutIntent] = None,
) -> tuple:
    """The extraction nets a budget MUST see to be measured honestly.

    A budget is only measurable if every net it reads is actually present in the
    extraction. If a referenced net is absent, the underlying metric silently
    treats that net's parasitics as 0.0 -- which would make the obligation report
    a misleading 0.0 imbalance ("pass") for a relationship that was never checked.
    The checker compares these against ``report.nets()`` and marks the obligation
    UNMEASURABLE instead.

      * ``matched_cc_balance`` reads each net's TOTAL cap, so BOTH nets of the
        pair must be present (an absent net is a phantom-0 total, not a balance).
      * ``kickback`` sums every aggressor's coupling onto each victim, so every
        VICTIM (or pair) net AND every aggressor must be present -- a missing
        aggressor would silently contribute 0 to every victim's sum, giving the
        same false-balance the all-pair-missing case gives.
      * ``clk_regen`` reads the (clock_net -> pair[0]) and (clock_net -> pair[1])
        coupling caps, so the resolved CLOCK net (``budget.clock_net`` or
        ``"clk"``) and BOTH pair nets must be present.
    """
    if budget.kind == "clk_regen":
        return (_budget_clock_net(budget),) + tuple(budget.pair)
    if budget.kind == "kickback":
        victims = (
            tuple(budget.victim_set)
            if budget.victim_set is not None
            else tuple(budget.pair)
        )
        aggressors = _budget_aggressors(budget, intent)
        # Aggressors that ARE the victim itself contribute 0 to that victim's
        # sum (the metric explicitly skips ``agg == victim``), so they need not
        # be present on the extraction to be honest -- only the OTHER aggressors
        # and every victim must exist.
        return victims + aggressors
    if budget.kind == "parasitic_cap" and parasitic_cap_is_single_ended(budget.pair):
        # A single-ended parasitic_cap(net, rail) reads the SIGNAL net's caps and
        # the (net, rail) coupling. Only the signal net must be present: an absent
        # rail simply means 0 cap-to-rail (off_rail == 1, an honest fail), not an
        # unmeasurable relationship -- so the signal net is the only requirement.
        signal, _rail = _single_ended_signal_and_rail(budget.pair)
        return (signal,)
    # matched_cc_balance / two-signal parasitic_cap need exactly their two pair nets.
    return tuple(budget.pair)


def _measure_budget(
    budget: ParasiticBudget,
    report: PexReport,
    intent: Optional[LayoutIntent] = None,
) -> Optional[float]:
    """Measured RELATIVE imbalance for one budget against the extraction.

    Returns ``None`` when the budget kind has no measurement mapping (a declared
    intent the metric layer has not yet implemented). Returning ``None`` -- rather
    than raising -- is the v2 contract that an unmapped budget becomes an
    ``unmeasurable`` obligation with the kind in ``missing_nets`` rather than a
    silent pass. Every CURRENTLY-mapped kind (matched_cc_balance / kickback /
    clk_regen) returns a float.
    """
    if budget.kind == "parasitic_cap" and parasitic_cap_is_single_ended(budget.pair):
        # SINGLE-ENDED parasitic_cap(net, rail): measured as the off-rail coupling
        # fraction, NOT the differential pair-imbalance metric. A signal net and a
        # supply/ground rail are asymmetric by design, so scoring them with the
        # matched-pair imbalance would produce a meaningless ~1.0 false-fail.
        return _measure_single_ended_parasitic_cap(report, tuple(budget.pair))
    if budget.kind in ("matched_cc_balance", "parasitic_cap"):
        # A two-SIGNAL-net ``parasitic_cap(a, b)`` is a genuine node-cap balance
        # between two symmetric signal nets, measured identically to
        # ``matched_cc_balance`` (per-pair total-cap relative imbalance,
        # max-denominator). Look up exactly this pair; fall back to direct
        # computation if the metric did not report it (e.g. nets absent).
        bal = differential_balance(report, pairs=(budget.pair,))
        per_pair = bal["pairs"]
        key = f"{budget.pair[0]}~{budget.pair[1]}"
        entry = per_pair.get(key)
        if entry is not None:
            return float(entry["imbalance"])
        ca = report.net_total_cap_aF(budget.pair[0])
        cb = report.net_total_cap_aF(budget.pair[1])
        denom = max(ca, cb, 1e-30)
        return abs(ca - cb) / denom
    if budget.kind == "kickback":
        aggressors = _budget_aggressors(budget, intent)
        return float(
            kickback_cc(report, aggressors=aggressors, inputs=tuple(budget.pair))[
                "imbalance"
            ]
        )
    if budget.kind == "clk_regen":
        return float(
            clk_regen_balance(
                report,
                clock_net=_budget_clock_net(budget),
                pair=tuple(budget.pair),
            )["imbalance"]
        )
    # Declared kind we do not yet measure: surface as unmeasurable.
    return None


def _check_lde_budgets(
    intent: LayoutIntent,
    geometry: Optional[LdeGeometry],
) -> List[LayoutObligation]:
    """Turn every :class:`LdeBudget` into a :class:`LayoutObligation`.

    Measures over DEVICE GEOMETRY (not the PEX caps): runs
    :func:`clay.layout.device_environment.extract_device_environment` ONCE on the
    supplied :class:`LdeGeometry`, then for each budget takes the two devices'
    ``metric`` and reduces it to the RELATIVE imbalance ``|va - vb| / max(va, vb)``.

    UNMEASURABLE -> token withheld (never a silent pass), in three cases, each
    surfaced with a ``missing_nets`` reason rather than a fabricated number:

      * NO geometry threaded to the checker (``geometry is None``);
      * a named DEVICE is absent from the extracted environment;
      * the metric is ``None`` for EITHER device (the required layer -- nwell /
        diffusion -- was absent from the extraction, so the extractor honestly
        reports ``None``).

    The obligation ``kind`` is ``lde:<metric>`` so it is distinguishable from a
    cap budget in the signoff rows while staying in the device-environment family.
    """
    # Map the budget metric name to the DeviceEnvironment dataclass attribute.
    _METRIC_ATTR = {
        "well_edge": "well_edge_distance",
        "od_length": "od_length",
        "od_neighbor_spacing": "od_neighbor_spacing",
    }
    lde_budgets = getattr(intent, "lde_budgets", ()) or ()
    if not lde_budgets:
        return []

    obligations: List[LayoutObligation] = []

    if geometry is None:
        # No geometry substrate reached the checker: every LDE budget is
        # unmeasurable (blocking), exactly like a missing net.
        for budget in lde_budgets:
            obligations.append(
                LayoutObligation(
                    kind=f"lde:{budget.metric}",
                    pair=budget.pair,
                    measured_imbalance=None,
                    rel_tol=budget.rel_tol,
                    missing_nets=("no_geometry",),
                )
            )
        return obligations

    from .device_environment import extract_device_environment

    env = extract_device_environment(
        dict(geometry.devices),
        list(geometry.layer_polygons),
        pdk_layers=geometry.pdk_layers,
        pmos_labels=(
            None if geometry.pmos_labels is None else list(geometry.pmos_labels)
        ),
    )

    for budget in lde_budgets:
        a, b = budget.pair
        # A named device missing from the environment cannot be measured.
        absent_devices = tuple(d for d in (a, b) if d not in env)
        if absent_devices:
            obligations.append(
                LayoutObligation(
                    kind=f"lde:{budget.metric}",
                    pair=budget.pair,
                    measured_imbalance=None,
                    rel_tol=budget.rel_tol,
                    missing_nets=tuple(f"device:{d}" for d in absent_devices),
                )
            )
            continue
        attr = _METRIC_ATTR[budget.metric]
        va = getattr(env[a], attr)
        vb = getattr(env[b], attr)
        # A None metric means the required layer was absent for that device ->
        # unmeasurable (NOT a fabricated 0.0), token withheld.
        none_sides = tuple(dev for dev, val in ((a, va), (b, vb)) if val is None)
        if none_sides:
            obligations.append(
                LayoutObligation(
                    kind=f"lde:{budget.metric}",
                    pair=budget.pair,
                    measured_imbalance=None,
                    rel_tol=budget.rel_tol,
                    missing_nets=tuple(f"{budget.metric}:{d}" for d in none_sides),
                )
            )
            continue
        denom = max(abs(float(va)), abs(float(vb)))
        if denom <= 0.0:
            # Both metrics are exactly zero -> perfectly matched (0 imbalance).
            measured = 0.0
        else:
            measured = abs(float(va) - float(vb)) / denom
        obligations.append(
            LayoutObligation(
                kind=f"lde:{budget.metric}",
                pair=budget.pair,
                measured_imbalance=measured,
                rel_tol=budget.rel_tol,
            )
        )
    return obligations


def _check_wire_width_intents(
    intent: LayoutIntent,
    routed_geometry: Optional[object],
) -> List[LayoutObligation]:
    """Turn declared per-net ``@wire_width`` widths into geometric obligations.

    For every net in ``intent.wire_widths`` (a ``@wire_width(net, w)`` declaration,
    width in nm) emit one ``wire_width`` :class:`LayoutObligation` measured DIRECTLY
    on the routed geometry by :func:`geometric_intent_checks.check_wire_width`: the
    net's realised wire width must be AT-OR-ABOVE the declared width (a RELATIVE
    under-width fraction reduced to ``measured_imbalance``, the shared pass test then
    being "realised >= declared within slack"). When NO routed geometry reached the
    checker but the cell DECLARED a width, the obligation is UNMEASURABLE (token
    withheld) -- a declared widening the router did not realise must BLOCK, never
    silently pass. A cell that declares no width emits no obligation (byte-identical
    verdict for every existing cell).
    """
    from .geometric_intent_checks import RoutedGeometry, check_wire_width

    wire_widths = dict(getattr(intent, "wire_widths", {}) or {})
    if not wire_widths:
        return []

    obligations: List[LayoutObligation] = []
    has_routed = isinstance(routed_geometry, RoutedGeometry)
    for net, declared in sorted(wire_widths.items()):
        if not has_routed:
            # Declared width, no routed geometry threaded in -> UNMEASURABLE/blocking
            # (the same policy a missing-substrate LDE budget gets).
            obligations.append(
                LayoutObligation(
                    kind="wire_width",
                    pair=(str(net), str(net)),
                    measured_imbalance=None,
                    rel_tol=_DEFAULT_WIRE_WIDTH_TOL,
                    missing_nets=(f"wire_width:no_routed_geometry:{net}",),
                )
            )
            continue
        measured, missing = check_wire_width(routed_geometry, str(net), float(declared))
        obligations.append(
            LayoutObligation(
                kind="wire_width",
                pair=(str(net), str(net)),
                measured_imbalance=measured,
                rel_tol=_DEFAULT_WIRE_WIDTH_TOL,
                missing_nets=missing,
            )
        )
    return obligations


def _check_net_spacing_intents(intent, routed_geometry) -> List[LayoutObligation]:
    """Require current GDS measurement; cached route boxes are insufficient."""
    from clay.layout.net_spacing import verify_exact_net_spacings
    declarations = tuple(getattr(intent, "net_spacings", ()))
    if not declarations:
        return []
    try:
        reports = verify_exact_net_spacings(
            getattr(routed_geometry, "exact_artifact", None),
            getattr(routed_geometry, "spacing_tech", None), declarations,
        )
        if len(reports) != len(declarations):
            raise ValueError("incomplete net_spacing measurement")
    except Exception as exc:
        return [LayoutObligation(
            kind="net_spacing", pair=(c.net_a, c.net_b), measured_imbalance=None, rel_tol=0.0,
            missing_nets=(f"net_spacing:unmeasurable:{type(exc).__name__}:{exc}",),
        ) for c in declarations]
    return [LayoutObligation(
        kind="net_spacing", pair=(c.net_a, c.net_b),
        # A binary constraint violation, not a claim of measured parasitic imbalance.
        measured_imbalance=None if r.status == "unmeasurable" else float(r.status != "pass"),
        rel_tol=0.0, missing_nets=tuple(r.missing_nets),
        measurements={"required_gap_nm": c.minimum_nm,
                      "violating_edge_pairs": sum(v[1] for v in r.violations)},
    ) for c, r in zip(declarations, reports)]


def _check_match_length_intents(
    intent: LayoutIntent,
    routed_geometry: Optional[object],
) -> List[LayoutObligation]:
    """Turn declared ``@match_length`` pairs into wire-length-balance obligations.

    For every ``(netA, netB)`` pair in ``intent.match_length_pairs`` (a
    ``@match_length(netA, netB)`` declaration) emit one ``match_length``
    :class:`LayoutObligation` measured DIRECTLY on the SAME routed geometry the
    ``wire_width`` obligation uses (per-net tagged routed metal shapes) by
    :func:`geometric_intent_checks.check_match_length`: the two nets' total routed
    wire-RUN lengths must be BALANCED within a RELATIVE imbalance slack
    (``|La - Lb| / max(La, Lb) <= rel_tol``). This function is the independent
    final-layout check; the router separately consumes the same pair as a hard
    Length constraint.

    When NO routed geometry reached the checker but the cell DECLARED a pair, the
    obligation is UNMEASURABLE (token withheld) -- a declared length-match the
    router did not realise must BLOCK, never silently pass. Likewise an absent net
    or a degenerate (zero-length) measurement yields UNMEASURABLE. A cell that
    declares no pair emits no obligation (byte-identical verdict for every existing
    cell). Net names are DATA (no per-net branch in this engine).
    """
    from .geometric_intent_checks import RoutedGeometry, check_match_length

    pairs = getattr(intent, "match_length_pairs", ())
    if type(pairs) is not tuple:
        raise TypeError("match_length_pairs must be an exact tuple")
    if not pairs:
        return []

    obligations: List[LayoutObligation] = []
    has_routed = isinstance(routed_geometry, RoutedGeometry)
    for pair in pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or not pair[0]
            or not pair[1]
            or pair[0] == pair[1]
        ):
            raise ValueError(
                "match_length declaration must contain two distinct net names"
            )
        net_a, net_b = pair
        if not has_routed:
            # Declared length-match, no routed geometry threaded in ->
            # UNMEASURABLE/blocking (the same policy wire_width / LDE budgets get).
            obligations.append(
                LayoutObligation(
                    kind="match_length",
                    pair=(net_a, net_b),
                    measured_imbalance=None,
                    rel_tol=_DEFAULT_MATCH_LENGTH_TOL,
                    missing_nets=(
                        f"match_length:no_routed_geometry:{net_a}",
                        f"match_length:no_routed_geometry:{net_b}",
                    ),
                )
            )
            continue
        measured, missing = check_match_length(routed_geometry, net_a, net_b)
        measurements = {}
        if measured is not None:
            length_a = routed_geometry.routed_length(net_a)
            length_b = routed_geometry.routed_length(net_b)
            if length_a is not None and length_b is not None:
                measurements = {
                    "route_length_a_dbu": float(length_a),
                    "route_length_b_dbu": float(length_b),
                }
        obligations.append(
            LayoutObligation(
                kind="match_length",
                pair=(net_a, net_b),
                measured_imbalance=measured,
                rel_tol=_DEFAULT_MATCH_LENGTH_TOL,
                missing_nets=missing,
                measurements=measurements,
            )
        )
    return obligations


def _check_geometric_intents(
    intent: LayoutIntent,
    placement_geometry: Optional[object],
    equal_size_groups: Sequence[Sequence[str]] = (),
    pinned_positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[LayoutObligation]:
    """Turn the STRUCTURAL placement intents into geometric obligations.

    Where the cap budgets measure against a PEX report and the LDE budgets against
    per-device well/OD geometry, the STRUCTURAL intents
    (``@interdigitated`` / common-centroid / ``@place_group``) are measured
    DIRECTLY on the placement geometry by
    :mod:`clay.layout.geometric_intent_checks`. Three obligation kinds are emitted:

      * ``interdig``             -- one per fused common-centroid cluster: the A/B
        fingers actually interleave, their sub-centroids coincide (relative tol),
        and every S/D column wires to the correct schematic net.
      * ``common_centroid``      -- one per matched device PAIR (sourced from the
        fused clusters' members): the pair's placed centroids coincide.
      * ``place_group_realized`` -- one per ``@place_group(row=, col=)`` directive:
        the group's devices land in the declared row/col band.

    When NO placement geometry reached the checker (``placement_geometry is None``)
    but the cell DECLARED structural intent (a fused cluster's provenance lives on
    the geometry, and ``intent.place_groups`` declares row/col bands), every such
    obligation is UNMEASURABLE (token withheld), exactly like a missing-substrate
    LDE budget -- never a silent pass.

    A cell that declared NO structural intent (no place_groups with row/col, no
    clusters) emits NO geometric obligations, so a non-structural cell's verdict is
    byte-identical to before.
    """
    from .geometric_intent_checks import (
        GeometricPlacement,
        check_abut,
        check_common_centroid,
        check_equal_size,
        check_fix,
        check_interdigitation,
        check_matching,
        check_order,
        check_place_group_realized,
    )

    obligations: List[LayoutObligation] = []

    # A coordinate-free place-group is a generation directive: it selects a
    # packing cluster but makes no final geometric claim.  Only row/col-bearing
    # groups create a realized-GDS obligation.
    place_groups = tuple(getattr(intent, "place_groups", ()) or ())
    structural_groups = [
        group
        for group in place_groups
        if place_group_floorplan_row(group) is not None
        or getattr(group, "col", None) is not None
    ]

    has_geom = isinstance(placement_geometry, GeometricPlacement)
    clusters = dict(placement_geometry.clusters) if has_geom else {}

    # NO placement geometry threaded in: the DIRECT geometric checks measure over
    # the PLACEMENT, so without it there is nothing to measure. We deliberately
    # emit NO obligations here rather than synthesise blocking ``no_geometry`` rows
    # -- the no-geometry block on a DECLARED structural intent (an interdigitation,
    # a common-centroid, a row/col place-group) is owned by the completeness gate
    # (``intent_completeness`` emits an ``unchecked_intent:<kind>`` row for any
    # declared structural intent with no covering obligation). Emitting our own
    # ``no_geometry`` rows here would double-block the same declaration AND inject
    # structural rows into a cap-only certify (which the cap-path callers do not
    # expect). The geometric obligations fire ONLY when geometry is supplied; the
    # cap-only path stays byte-identical to before.
    if not has_geom:
        return obligations

    # ── direct @matching obligations ────────────────────────────────────────
    # Reciprocal source declarations are already canonicalized on the intent;
    # emit exactly one final-geometry obligation and retain every directed ID.
    for matching in tuple(getattr(intent, "matching_intents", ()) or ()):
        if (
            matching.requires_interdigitation
            and matching.declaration_ids
            == matching.interdigitation_declaration_ids
        ):
            continue
        verdict = check_matching(
            placement_geometry,
            matching.members,
            rel_tol=_DEFAULT_CC_TOL,
            require_interdigitation=False,
        )
        measurements = {
            key: float(value)
            for key, value in (
                ("centroid_axis_residual", verdict.centroid_axis_residual),
                ("orientation_imbalance", verdict.orientation_imbalance),
                ("footprint_imbalance", verdict.footprint_imbalance),
            )
            if value is not None
        }
        obligations.append(
            LayoutObligation(
                kind="matching",
                pair=matching.members,
                measured_imbalance=verdict.measured,
                rel_tol=_DEFAULT_CC_TOL,
                missing_nets=verdict.missing,
                verification_status=VerificationStatus(verdict.status),
                declaration_ids=matching.declaration_ids,
                measurements=measurements,
            )
        )

    # ── interdig + common_centroid (one obligation each per fused cluster) ────
    # The fused clusters carry the DEVICE-level matched pairs (device_a/device_b)
    # AND the realised interleave, so they are the source of both the interdig and
    # the common_centroid (matched-pair) obligations. When geometry is absent but
    # the cell has structural place-groups, we still cannot enumerate the declared
    # clusters here (their provenance only exists post-placement); the place-group
    # obligations below carry the blocking-on-missing-geometry instead.
    for cluster in clusters.values():
        verdict = check_interdigitation(cluster)
        pair = (cluster.device_a, cluster.device_b)
        missing = (
            ()
            if verdict.measured is not None
            else (f"interdig:{verdict.reason or 'unmeasurable'}",)
        )
        obligations.append(
            LayoutObligation(
                kind="interdig",
                pair=pair,
                measured_imbalance=verdict.measured,
                rel_tol=_DEFAULT_INTERDIG_TOL,
                missing_nets=missing,
            )
        )
        # A matched/interdigitated pair is by construction a common-centroid group:
        # a DIRECT centroid-coincidence obligation over its two members.
        cc_measured, cc_missing = check_common_centroid(placement_geometry, pair)
        obligations.append(
            LayoutObligation(
                kind="common_centroid",
                pair=pair,
                measured_imbalance=cc_measured,
                rel_tol=_DEFAULT_CC_TOL,
                missing_nets=cc_missing,
            )
        )

    # ── place_group realisation (one obligation per declared row/col group) ───
    all_group_tuples = [
        (tuple(g.devices), place_group_floorplan_row(g), g.col)
        for g in structural_groups
    ]
    for g in structural_groups:
        # The obligation pair is consumed as an exactly-2-tuple by the signoff row
        # builder AND as the coverage-target set by the completeness gate
        # (``intent_completeness._covers`` matches a place_group obligation to its
        # declaration by device-set subset). Carry up to two of the group's DEVICE
        # labels so the gate recognises THIS obligation as covering THIS group; a
        # single-device group pads the group name into slot 2.
        gdevs = tuple(g.devices)
        if len(gdevs) >= 2:
            pair = (gdevs[0], gdevs[1])
        elif len(gdevs) == 1:
            pair = (gdevs[0], str(g.group))
        else:
            pair = (str(g.group), f"row{g.row_min}_col{g.col}")
        measured, missing = check_place_group_realized(
            placement_geometry,
            tuple(g.devices),
            all_groups=all_group_tuples,
            row=place_group_floorplan_row(g),
            col=g.col,
        )
        obligations.append(
            LayoutObligation(
                kind="place_group_realized",
                pair=pair,
                measured_imbalance=measured,
                rel_tol=_DEFAULT_PLACE_GROUP_TOL,
                missing_nets=missing,
            )
        )

    # ── equal_size realisation (one obligation per declared @equal_size group) ─
    # A declared @equal_size(d1, d2, ...) group's members must realise an
    # IDENTICAL footprint (same placed width AND height) within a small RELATIVE
    # spread. Measured DIRECTLY on the placed bboxes; UNMEASURABLE (blocking) when a
    # member has no realised bbox -- never a silent pass. The groups come from the
    # device-level @equal_size annotation (carried through ``SolverConstraints``),
    # not the net-level LayoutIntent, so they are threaded in explicitly.
    for grp in equal_size_groups:
        members = tuple(str(m) for m in grp)
        if len(members) < 2:
            continue
        measured, missing = check_equal_size(
            placement_geometry, members, rel_tol=_DEFAULT_EQUAL_SIZE_TOL
        )
        # The obligation pair is read as an exactly-2-tuple by the signoff row
        # builder + the completeness gate (which matches by device-set overlap).
        # Carry the first two members so the gate recognises THIS obligation as
        # covering THIS @equal_size declaration.
        pair = (members[0], members[1])
        obligations.append(
            LayoutObligation(
                kind="equal_size",
                pair=pair,
                measured_imbalance=measured,
                rel_tol=_DEFAULT_EQUAL_SIZE_TOL,
                missing_nets=missing,
            )
        )

    # ── @order realisation (one obligation per declared @order(a, b, rel)) ────────
    # The realised centroids must respect the declared relative order (left_of /
    # above). Measured DIRECTLY on the placed centroids; UNMEASURABLE (blocking) when
    # a member has no centroid -- never a silent pass. ``order_constraints`` is carried
    # on the LayoutIntent (cell-level), so it is read off ``intent`` here.
    for a, b, rel in tuple(getattr(intent, "order_constraints", ()) or ()):
        measured, missing = check_order(
            placement_geometry, str(a), str(b), str(rel), rel_tol=_DEFAULT_ORDER_TOL
        )
        obligations.append(
            LayoutObligation(
                kind="order",
                pair=(str(a), str(b)),
                measured_imbalance=measured,
                rel_tol=_DEFAULT_ORDER_TOL,
                missing_nets=missing,
            )
        )

    # ── @abut realisation (one obligation per declared @abut(a, b)) ───────────────
    # The two devices' facing-edge gap must be at (near) the DRC-minimum. Measured
    # DIRECTLY on the placed bboxes; UNMEASURABLE (blocking) when a member has no
    # bbox or the two overlap with no facing edge -- never a silent pass.
    for a, b in tuple(getattr(intent, "abut_pairs", ()) or ()):
        measured, missing = check_abut(
            placement_geometry, str(a), str(b), rel_tol=_DEFAULT_ABUT_TOL
        )
        obligations.append(
            LayoutObligation(
                kind="abut",
                pair=(str(a), str(b)),
                measured_imbalance=measured,
                rel_tol=_DEFAULT_ABUT_TOL,
                missing_nets=missing,
            )
        )

    # ── @fix realisation (one obligation per declared @fix(device, x, y)) ──────────
    # The pinned device's realised centroid must sit within one grid quantum of the
    # absolute pin. Measured DIRECTLY on the placed centroid; UNMEASURABLE (BLOCKING)
    # when the device has no realised bbox -- never a silent pass. ``pinned_positions``
    # is a DEVICE-level annotation carried through ``SolverConstraints`` (microns), so it
    # is threaded in explicitly (like ``equal_size_groups``), not read off the LayoutIntent.
    for device, (x_um, y_um) in dict(pinned_positions or {}).items():
        pin_x_nm = float(x_um) * _FIX_UM_TO_NM
        pin_y_nm = float(y_um) * _FIX_UM_TO_NM
        measured, missing = check_fix(
            placement_geometry, str(device), pin_x_nm, pin_y_nm
        )
        obligations.append(
            LayoutObligation(
                kind="fix",
                pair=(str(device), str(device)),
                measured_imbalance=measured,
                rel_tol=_DEFAULT_FIX_TOL,
                missing_nets=missing,
            )
        )

    return obligations


def check_layout_intent(
    intent: LayoutIntent,
    report: PexReport,
    *,
    geometry: Optional[LdeGeometry] = None,
    placement_geometry: Optional[object] = None,
    routed_geometry: Optional[object] = None,
    equal_size_groups: Sequence[Sequence[str]] = (),
    pinned_positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[LayoutObligation]:
    """Check every parasitic budget of ``intent`` against ``report``.

    Returns one :class:`LayoutObligation` per :class:`ParasiticBudget`, carrying
    the measured RELATIVE imbalance and the budget's ``rel_tol``. The obligation's
    own ``.status`` / ``.over_budget`` derive the pass/fail verdict.

    UNMEASURABLE budgets: before measuring, the checker confirms every net the
    budget needs (see :func:`_required_nets`) is present in ``report.nets()``. If
    a needed net is ABSENT from the extraction the budget cannot be measured
    honestly -- the underlying metric would attribute 0.0 cap to the missing net
    and report a misleading 0.0 imbalance ("pass") for a relationship that was
    never actually extracted. Such a budget yields an obligation with
    ``measured_imbalance=None`` (``status == "unmeasurable"``) naming the
    ``missing_nets``, rather than a false pass/fail. The verdict treats it as
    blocking (it is not a pass).

    NET RESOLUTION: a declared net includes both its exact root and extracted
    ``.n#``/``.t#``/``.#`` segments. Its capacitance is the signed sum across that
    conductor's boundary, excluding internal capacitance. Missing members,
    conflicting ownership, or a resistor crossing the membership boundary make
    the affected budget UNMEASURABLE. Presence is checked against the original
    membership so a conductor with zero external capacitance remains measurable.
    """
    resolved, resolution = resolve_report_for_intent(report, intent, intent.budgets)
    obligations: List[LayoutObligation] = []
    for budget in intent.budgets:
        required = _required_nets(budget, intent)
        missing = tuple(n for n in required if not _resolve_net(n, resolution))
        # A single-ended rail may be absent (all coupling is then off-rail),
        # but an extracted rail with invalid membership cannot be measured.
        missing += tuple(
            n for n in budget.pair
            if n not in required and resolution.get(n.lower()) == ()
        )
        if missing:
            obligations.append(
                LayoutObligation(
                    kind=budget.kind,
                    pair=budget.pair,
                    measured_imbalance=None,
                    rel_tol=budget.rel_tol,
                    missing_nets=missing,
                )
            )
            continue
        measured = _measure_budget(budget, resolved, intent)
        if measured is None:
            # Declared budget kind the metric layer has no mapping for: surface
            # as unmeasurable with the kind itself in missing_nets so the row
            # explains WHY it could not be measured (no_mapping), rather than
            # silently passing.
            obligations.append(
                LayoutObligation(
                    kind=budget.kind,
                    pair=budget.pair,
                    measured_imbalance=None,
                    rel_tol=budget.rel_tol,
                    missing_nets=(f"no_mapping:{budget.kind}",),
                )
            )
            continue
        obligations.append(
            LayoutObligation(
                kind=budget.kind,
                pair=budget.pair,
                measured_imbalance=measured,
                rel_tol=budget.rel_tol,
            )
        )
    # Device-environment (LDE) budgets measure over device GEOMETRY, not the PEX
    # caps; appended after the cap budgets so the cap-budget path above is
    # byte-identical to before. An LDE budget with no geometry threaded through
    # is unmeasurable (blocking), never a silent pass.
    obligations.extend(_check_lde_budgets(intent, geometry))
    # Structural placement intents (interdigitation / common-centroid /
    # place_group) measure over the PLACEMENT geometry, not the PEX caps; appended
    # after the cap + LDE budgets so those paths stay byte-identical. A declared
    # structural intent with no placement geometry threaded through is unmeasurable
    # (blocking), never a silent pass.
    obligations.extend(
        _check_geometric_intents(
            intent, placement_geometry, equal_size_groups, pinned_positions
        )
    )
    # Declared per-net wire WIDTHS measure over the ROUTED geometry (not the PEX
    # caps / placement): appended last so every other path stays byte-identical. A
    # declared `@wire_width` with no routed geometry threaded through is unmeasurable
    # (blocking), never a silent pass.
    obligations.extend(_check_wire_width_intents(intent, routed_geometry))
    # Declared matched-net wire-LENGTH balance pairs measure over the SAME ROUTED
    # geometry as the wire WIDTH obligation; appended last so every other path stays
    # byte-identical. A declared `@match_length` with no routed geometry threaded
    # through is unmeasurable (blocking), never a silent pass. This checker does
    # not route; route compilation separately consumes the same canonical pair.
    obligations.extend(_check_match_length_intents(intent, routed_geometry))
    obligations.extend(_check_net_spacing_intents(intent, routed_geometry))
    return obligations


def layout_intent_verdict(obligations: List[LayoutObligation]) -> Dict[str, object]:
    """Aggregate a list of obligations into a single pass/fail verdict.

    ``passed`` is True iff every obligation is a measured PASS (an unmeasurable
    obligation has status != "pass", so it blocks). With zero obligations it is
    vacuously True -- read the ``empty`` flag to distinguish that from a verified
    cell. ``failed`` lists the obligations whose measured imbalance is over budget.
    ``unmeasurable`` lists the obligations that could not be measured (a needed
    net was absent from the extraction); these BLOCK the verdict exactly like a
    pending obligation -- an unmeasurable relationship was never actually checked,
    so the cell cannot be called verified. ``worst`` is the obligation with the
    largest ``over_budget`` among the MEASURED ones (None when none were measured).

    EMPTY budgets: a cell that declared no parasitic budgets (e.g. a single-ended
    block with no diff pairs) is vacuously passing, but the verdict flags it with
    ``empty=True`` so a cell that declared NO intent is never silently mistaken
    for one whose intent was verified. ``empty`` is False whenever at least one
    obligation exists.
    """
    failed = [o for o in obligations if o.status == VerificationStatus.FAIL.value]
    unmeasurable = [
        o for o in obligations if o.status == VerificationStatus.UNMEASURABLE.value
    ]
    unsupported = [
        o for o in obligations if o.status == VerificationStatus.UNSUPPORTED.value
    ]
    measured = [o for o in obligations if o.measured_imbalance is not None]
    worst = max(measured, key=lambda o: o.over_budget, default=None)
    return {
        # Passed iff every obligation measured within budget. An unmeasurable
        # obligation has status != "pass", so it blocks here exactly like a
        # pending obligation. An EMPTY budget set is vacuously passing -- the
        # ``empty`` flag (not a forced False) is what keeps a no-intent cell from
        # being read as a verified one.
        "passed": all(o.status == VerificationStatus.PASS.value for o in obligations),
        "failed": failed,
        "unmeasurable": unmeasurable,
        "unsupported": unsupported,
        "worst": worst,
        "empty": not obligations,
    }


def check_shield_intent(intent: LayoutIntent, cell) -> List[Dict[str, object]]:
    """Check the geometric shielding requirements of ``intent`` against a routed cell.

    Secondary to the PEX budgets: for each declared :class:`Shield`, ask
    ``comparator_metrics.shielding_between`` whether a grounded shape actually sits
    between the aggressor and victim. Returns one plain dict per shield with the
    requirement, the ``shielded`` verdict, and the supporting geometry. Requires a
    routed ``cell`` (with ``.tagged`` shapes); pass the same cell whose PEX report
    was checked.
    """
    out: List[Dict[str, object]] = []
    for shield in intent.shields:
        res = shielding_between(
            cell, shield.aggressor, shield.victim, gnd_net=shield.shield_net
        )
        out.append(
            {
                "aggressor": shield.aggressor,
                "victim": shield.victim,
                "shield_net": shield.shield_net,
                "shielded": bool(res.get("shielded")),
                "geometry": res,
            }
        )
    return out
