"""Phase P -- the PLACEMENT convergence loop for the Clay intent-convergent
place + route engine.

Directive (design doc, 2026-05-29): do NOT place then immediately route. Place,
then iterate the placement against the declared INTENT until it converges; only
then route. This module is Phase P: it converges a cell's placement to its
designer intent (the per-cell ``ConvergenceCriteria`` lifted from the
``layout_guide.txt`` + ``layout_intent {}``) AND to DRC-legality, by re-solving
the generic MILP placer with adjusted constraints until the worst unmet
obligation is satisfied.

What it COMPOSES (imported read-only -- never edited here)
=========================================================
* :func:`clay.layout.generic_placement.place_generic` -- the generic MILP placer
  that turns a parsed ``CellDef`` (+ its ``LayoutIntent``, + the bound ``.cir``
  sizing) into a :class:`Placement` with mirror-symmetric matched pairs about a
  solved vertical axis. The loop re-invokes it with adjusted knobs each iteration.
* :func:`clay.layout.annotate_to_solver.score_placement` -- the REAL R172
  symmetry / common-centroid / rail quality, used (via the placement's own
  ``quality``) as the fallback acceptance when no external ``criteria`` is given.
* :func:`clay.layout.drc_legal_spacing.inter_device_min_gap` /
  :func:`required_inter_device_spacing_nm` (gf180 ``Gf180mcuLayerMap``) -- the
  placement-level DRC-legality check: every separated adjacent-device pair must
  meet the PDK inter-device min-spacing rule.

Acceptance (per the round + design)
-----------------------------------
A placement is CONVERGED iff the intent obligations are satisfied
(``criteria.check_placement(placement).satisfied`` -- symmetry / centroid / rows)
AND the placement is DRC-legal (no device-spacing / overlap violation). It is NOT
a geometric-cost optimum: signoff intent + DRC ARE the stopping condition.

Generic adjustment (no per-circuit branches)
--------------------------------------------
Each iteration measures the worst UNMET obligation and adjusts ONE generic knob
toward it, then re-solves via :func:`place_generic`:

* the worst unmet is a DRC / spacing legality miss  -> WIDEN spacing (coarsen the
  placement grid, which uniformly grows every inter-device gap);
* the worst unmet is an INTENT obligation (symmetry / centroid residual) -> TIGHTEN
  the symmetry acceptance tolerance the scorer / criteria use, re-solving the MILP
  against a stricter mirror requirement.

The adjustment is driven entirely by the MEASURED worst obligation, identical for
every circuit; an obligation that cannot be satisfied within ``max_iters`` is
reported (``converged=False`` + the residual + the unmet obligation) rather than
looped forever.

Honesty contract
----------------
* The placement is the REAL MILP solver output every iteration; the symmetry /
  centroid residual is the REAL R172 measurement; the DRC-legality is the REAL
  gf180 inter-device spacing rule. Nothing is fabricated per circuit.
* The four production circuits are PROBES that grade convergence; this engine
  improves, never a circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from clay.ir import CellDef
from clay.layout.generic_placement import Placement, place_generic
from clay.layout.intent import LayoutIntent

# Placement-level DRC-legality (real gf180 inter-device min-spacing rule).
from clay.layout.backend import Gf180mcuLayerMap
from clay.layout.drc_legal_spacing import (
    inter_device_min_gap,
    required_inter_device_spacing_nm,
)

Bbox = Tuple[int, int, int, int]

# --- generic adjustment knobs ------------------------------------------------
#
# The loop adjusts these (and re-solves via place_generic) toward the worst unmet
# obligation. They are GENERIC -- one schedule for every circuit, no per-cell
# table. The starting values match place_generic's own defaults so the first
# iteration is the baseline placement.
_GRID_NM_START = 5
_GRID_NM_STEP = 5          # coarsen the grid by 5nm each DRC-driven adjustment
_GRID_NM_MAX = 40          # cap (a too-coarse grid stops helping spacing)
_REL_TOL_START = 0.05      # the R172 symmetry acceptance tolerance
_REL_TOL_MIN = 0.005       # floor (tightening below this stops being meaningful)
_REL_TOL_FACTOR = 0.5      # halve the tolerance each intent-driven adjustment


# --- result + history --------------------------------------------------------


@dataclass(frozen=True)
class PlacementIteration:
    """One iteration of the placement convergence loop (audit trail)."""

    index: int
    symmetry: float
    intent_satisfied: bool
    drc_legal: bool
    residual: float
    worst_obligation: str
    action: str
    grid_nm: int
    rel_tol: float
    min_gap_nm: Optional[int]
    required_gap_nm: int


@dataclass(frozen=True)
class ConvergedPlacement:
    """The Phase-P result: the converged (or best-effort) placement + provenance.

    ``placement`` is the final :class:`Placement`. ``converged`` is True iff the
    intent obligations were satisfied AND the placement is DRC-legal within
    ``max_iters``. ``history`` is the per-iteration audit (symmetry / residual /
    worst obligation / action taken). ``residual`` is the final
    distance-to-satisfaction (0.0 when converged). ``unmet`` names the worst
    obligation still unsatisfied when ``converged`` is False (empty otherwise).
    """

    placement: Placement
    iters: int
    converged: bool
    history: Tuple[PlacementIteration, ...]
    residual: float = 0.0
    unmet: str = ""
    note: str = ""

    @property
    def symmetry_score(self) -> float:
        return float(self.placement.symmetry_score)


# --- intent acceptance: external criteria OR the score_placement fallback ----


@dataclass(frozen=True)
class _FallbackResult:
    """A :class:`CriteriaResult`-shaped acceptance built from ``score_placement``
    when no external ``criteria`` is supplied, so the loop is testable standalone.
    Mirrors the frozen ``CriteriaResult`` interface (``satisfied`` / ``obligations``
    / ``worst`` / ``reasons``)."""

    satisfied: bool
    obligations: tuple
    worst: float
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class _FallbackCriteria:
    """The default acceptance when no Agent-A ``ConvergenceCriteria`` is wired in.

    Accepts a placement iff its REAL R172 symmetry score (from
    ``score_placement``, carried on ``Placement.quality``) meets a threshold. This
    is the symmetry obligation only -- the DRC-legality gate is applied separately
    by the loop -- so the loop converges + reports standalone against the real
    measured symmetry without needing the full guide-derived criteria.
    """

    cell: str = ""
    symmetry_threshold: float = 0.99

    def check_placement(self, placement: Placement) -> _FallbackResult:
        score = float(getattr(placement, "symmetry_score", 0.0))
        satisfied = score >= self.symmetry_threshold
        # distance-to-satisfaction: how far the symmetry residual is from the
        # threshold (0.0 when met). This is the adjoint the loop minimises.
        residual = max(0.0, self.symmetry_threshold - score)
        reason = (
            f"symmetry {score:.4f} >= {self.symmetry_threshold:.4f}"
            if satisfied
            else f"symmetry {score:.4f} < {self.symmetry_threshold:.4f} "
            f"(residual {residual:.4f})"
        )
        return _FallbackResult(
            satisfied=satisfied,
            obligations=(("symmetry", score, satisfied),),
            worst=residual,
            reasons=(reason,),
        )


# --- placement-level DRC-legality (real gf180 inter-device spacing rule) ------


@dataclass(frozen=True)
class _DrcResult:
    legal: bool
    min_gap_nm: Optional[int]
    required_nm: int
    deficit_nm: int
    reason: str


def _placement_drc_legal(placement: Placement, pdk_layers=None) -> _DrcResult:
    """Placement-level DRC-legality: every separated adjacent-device pair must
    meet the PDK inter-device min-spacing rule (poly-poly / diff-diff).

    Composes :func:`inter_device_min_gap` over the placed device bboxes against
    :func:`required_inter_device_spacing_nm` (gf180 by default). This is a
    placement-level legality gate (device spacing), NOT a full geometric DRC --
    the full deck runs in Phase R on the routed GDS. An empty / single-device
    placement is vacuously legal.
    """
    if pdk_layers is None:
        pdk_layers = Gf180mcuLayerMap()
    required = int(required_inter_device_spacing_nm(pdk_layers))
    placed: Dict[str, Bbox] = {
        lbl: pd.bbox for lbl, pd in placement.devices.items()
    }
    if len(placed) < 2:
        return _DrcResult(True, None, required, 0, "fewer than two devices")
    tightest = inter_device_min_gap(placed)
    if tightest is None:
        return _DrcResult(True, None, required, 0, "no separated device pairs")
    gap, a, b, axis = tightest
    if gap >= required:
        return _DrcResult(
            True, gap, required, 0,
            f"tightest gap {gap}nm >= {required}nm ({a}/{b} on {axis})",
        )
    deficit = required - gap
    return _DrcResult(
        False, gap, required, deficit,
        f"tightest gap {gap}nm < {required}nm ({a}/{b} on {axis}); "
        f"deficit {deficit}nm",
    )


# --- placement view the criteria can read ------------------------------------


class _CriteriaPlacementView:
    """A thin read-only adapter that exposes a :class:`Placement` in the shape an
    external ``ConvergenceCriteria.check_placement`` reads device centroids from.

    The Agent-A criteria reads centroids via a ``device_centroids()`` method /
    ``.device_centroids`` mapping (or the router's native ``.placement.term``
    form). The generic placer's :class:`Placement` instead carries a
    ``{label: PlacedDevice}`` map with a ``.bbox`` per device. This adapter is the
    GENERIC bridge between the two frozen interfaces (it does not edit either
    side): it computes each device's centroid as its placed-bbox centre and
    re-exposes the placement's other attributes (``net_devices`` / ``axis_x`` /
    quality) by delegation, so the criteria measures the REAL placed geometry.

    Bridging here (in the loop, the consumer of both interfaces) keeps the placer
    and the criteria each unmodified -- the placement shape and the criteria's
    centroid contract are reconciled by the engine that composes them.
    """

    def __init__(self, placement: Placement):
        self._p = placement

    def device_centroids(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for lbl, pd in self._p.devices.items():
            x0, y0, x1, y1 = pd.bbox
            out[str(lbl)] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return out

    def __getattr__(self, name):
        # delegate every other access (axis_x, geom, quality, net_devices, ...)
        # to the wrapped placement so the criteria sees the real placement state.
        return getattr(self._p, name)


# --- the convergence loop -----------------------------------------------------


def _check_intent(criteria, placement: Placement):
    """Run the intent acceptance (external criteria's ``check_placement`` or the
    fallback). Returns a CriteriaResult-shaped object with ``satisfied`` /
    ``worst`` / ``reasons``.

    The fallback criteria reads ``Placement.symmetry_score`` directly; an external
    ``ConvergenceCriteria`` reads device centroids, so the placement is presented
    through :class:`_CriteriaPlacementView` (which exposes ``device_centroids()``
    from the placed bboxes) when the criteria is not the built-in fallback."""
    if isinstance(criteria, _FallbackCriteria):
        return criteria.check_placement(placement)
    return criteria.check_placement(_CriteriaPlacementView(placement))


# Generic obligation-name -> adjustment-lever map. Obligation names are the
# Agent-A frozen contract ("sym[...]", "cc[...]", "compact[...]"); the prefix
# alone (before the bracket) classifies the obligation. A mirror-symmetry miss is
# movable by tightening the symmetry acceptance; a common-centroid / compactness
# miss needs a centroid-placement constraint this engine's MILP does not yet
# express, so it is reported as an engine-capability gap rather than burned on an
# ineffective lever. One uniform map for every circuit, no per-cell branch.
_INTENT_LEVER = {
    "sym": "tighten_symmetry",
}
# obligation prefixes with NO lever in this engine (reported, not iterated).
_UNMOVABLE_INTENT = {"cc", "compact"}


def _intent_obligation_prefix(name: str) -> str:
    """The obligation class prefix (the token before ``[``): ``sym`` / ``cc`` /
    ``compact`` / the whole name when it has no bracket."""
    return name.split("[", 1)[0] if "[" in name else name


def _worst_unmet_intent_obligation(intent_res) -> Tuple[str, float]:
    """The worst UNMET intent obligation as ``(prefix, residual_over_tol)``.

    Reads the criteria's ``obligations`` rows ``(name, ok, measured, tol)`` and
    returns the failing obligation with the largest residual-over-tolerance, so
    the lever is chosen for the obligation actually dominating the miss. Falls
    back to ``("", worst)`` when the criteria exposes no per-obligation rows."""
    rows = getattr(intent_res, "obligations", ()) or ()
    worst_prefix = ""
    worst_resid = -1.0
    for row in rows:
        try:
            name, ok, measured, tol = row
        except (ValueError, TypeError):
            continue
        if ok:
            continue
        resid = (
            float(measured) - float(tol)
            if measured is not None
            else float(getattr(intent_res, "worst", 0.0))
        )
        if resid > worst_resid:
            worst_resid = resid
            worst_prefix = _intent_obligation_prefix(str(name))
    if worst_prefix == "":
        return ("", float(getattr(intent_res, "worst", 0.0)))
    return (worst_prefix, max(0.0, worst_resid))


def _worst_obligation(
    intent_res, drc: _DrcResult
) -> Tuple[str, float, str]:
    """Pick the worst UNMET obligation across intent + DRC, generically.

    Returns ``(kind, residual, action)`` where ``kind`` is the obligation class
    ("intent" / "drc" / "" if all met), ``residual`` is its distance-to-sat, and
    ``action`` is the generic adjustment to apply. The worst obligation is the one
    with the larger NORMALISED residual: the DRC deficit normalised by the required
    spacing, vs the intent residual. The intent action is chosen from the FAILING
    obligation's class (``sym`` -> tighten symmetry; ``cc`` / ``compact`` -> no
    engine lever -> ``report_gap``), so an ineffective knob is never burned.
    """
    intent_ok = bool(getattr(intent_res, "satisfied", False))
    intent_resid = float(getattr(intent_res, "worst", 0.0))
    drc_resid = (
        (drc.deficit_nm / drc.required_nm) if (not drc.legal and drc.required_nm)
        else 0.0
    )
    if intent_ok and drc.legal:
        return ("", 0.0, "none")

    def _intent_decision() -> Tuple[str, float, str]:
        prefix, resid = _worst_unmet_intent_obligation(intent_res)
        if prefix in _UNMOVABLE_INTENT:
            # no lever in this engine for a centroid / compactness placement
            # obligation -> report it (the loop stops on report_gap).
            return ("intent", resid, "report_gap")
        action = _INTENT_LEVER.get(prefix, "tighten_symmetry")
        return ("intent", resid if prefix else intent_resid, action)

    # both can be unmet; adjust toward the larger normalised residual.
    if not drc.legal and drc_resid >= intent_resid:
        return ("drc", drc_resid, "widen_spacing")
    if not intent_ok:
        return _intent_decision()
    # DRC unmet but its residual is smaller than a (met) intent's -> still DRC.
    return ("drc", drc_resid, "widen_spacing")


def _clamp_grid_nm(value) -> int:
    """Clamp a seeded ``grid_nm`` to the engine's valid range, rounded to int."""
    return int(max(_GRID_NM_START, min(_GRID_NM_MAX, round(float(value)))))


def _clamp_rel_tol(value) -> float:
    """Clamp a seeded ``rel_tol`` to the engine's valid [floor, start] range."""
    return float(max(_REL_TOL_MIN, min(_REL_TOL_START, float(value))))


def _place_fingerprint(cell_def, effective_policy, intent, pdk):
    """Build the place-phase intent-context fingerprint, or None on any failure.

    Composes :func:`clay.layout.learning_memory.intent_context_fingerprint`. A
    fingerprint failure (an unusual intent / policy shape) must never break the
    placement loop -- learning degrades to OFF, the cold run proceeds unchanged.
    """
    try:
        from clay.layout.learning_memory import intent_context_fingerprint

        cell_name = str(getattr(cell_def, "local_name", "") or "cell")
        pdk_name = str(pdk) if pdk else "gf180mcu"
        return intent_context_fingerprint(
            cell_name, effective_policy, intent, phase="place", pdk=pdk_name
        )
    except Exception:
        return None


def place_to_intent(
    cell_def: CellDef,
    intent: Optional[LayoutIntent] = None,
    criteria=None,
    pdk=None,
    *,
    source_cir=None,
    drc=None,
    max_iters: int = 6,
    pdk_layers=None,
    memory=None,
    enable_learning: bool = False,
    effective_policy=None,
    engine: str = "milp",
    style_reference=None,
    tech=None,
    native_geometry=None,
    native_composite_row=None,
) -> ConvergedPlacement:
    """Phase P: converge a cell's PLACEMENT to its declared intent + DRC-legality.

    1. ``placement = place_generic(cell_def, intent, source_cir=source_cir)``.
    2. loop up to ``max_iters``:
         * ``res = criteria.check_placement(placement)`` (symmetry / centroid /
           rows; the fallback uses the real R172 symmetry threshold when
           ``criteria is None``);
         * placement DRC-legality (gf180 inter-device min-spacing);
         * if ``res.satisfied`` AND DRC-legal -> CONVERGED, break;
         * else find the worst unmet obligation and adjust ONE generic knob toward
           it (widen spacing for DRC, tighten symmetry tol for intent), then
           re-solve via ``place_generic``.
    3. return :class:`ConvergedPlacement` (placement, iters, converged, history).

    The acceptance is intent (``criteria.check_placement``) AND DRC-legality -- not
    geometric cost. The adjustment is generic from the measured worst obligation;
    an unsatisfiable obligation is reported (``converged=False`` + residual + unmet
    obligation), never infinite-looped.

    ``criteria`` may be an Agent-A ``ConvergenceCriteria`` (consumed via its frozen
    ``check_placement`` interface) or ``None`` (the symmetry-threshold fallback).
    ``drc`` is reserved for a caller-supplied placement-DRC predicate; when ``None``
    the built-in gf180 inter-device spacing gate is used.

    v3 learning memory (``memory`` + ``enable_learning``): when a
    :class:`~clay.layout.learning_memory.LayoutMemory` is supplied AND
    ``enable_learning`` is True AND an ``effective_policy`` is given, the loop
    builds the place-phase intent-context fingerprint, SEEDS ``grid_nm`` /
    ``rel_tol`` from the blend of prior converged runs (clamped to the engine's
    valid ranges) when a match exists, and RECORDS the run's outcome afterwards.
    When ``memory is None`` (the default) the behaviour is byte-for-byte unchanged
    -- no fingerprint is built, no seed is consulted, nothing is recorded.
    """
    if criteria is None:
        criteria = _FallbackCriteria(cell=getattr(cell_def, "local_name", ""))

    grid_nm = _GRID_NM_START
    rel_tol = _REL_TOL_START

    # v3: seed the knobs from accumulated experience (default OFF). Learning needs
    # both an effective_policy (for the fingerprint's strategy fields) and an
    # intent; when either is absent we skip learning gracefully (no fingerprint).
    _fp = None
    if memory is not None and enable_learning and effective_policy is not None:
        _fp = _place_fingerprint(cell_def, effective_policy, intent, pdk)
        if _fp is not None:
            seed = memory.seed(_fp)
            if seed:
                if "grid_nm" in seed:
                    grid_nm = _clamp_grid_nm(seed["grid_nm"])
                if "rel_tol" in seed:
                    rel_tol = _clamp_rel_tol(seed["rel_tol"])

    # Placement engine dispatch. "milp" (default) = the exact symmetry-constrained
    # solver (knob-adjustable). "seqpair" = the constraint-graph placer (overlap-free
    # by construction, scalable; grid_nm/rel_tol knobs do not apply, so the loop
    # converges in one solve or reports the common-centroid gap). A ``style_reference``
    # (a RelativeOrder) transfers a reference layout's order before solving (B).
    # Resolve the placement technology ONCE (gf180mcu by default -- parity + no
    # caller breaks). ``tech`` (P3+) takes precedence; a legacy ``pdk`` is resolved
    # to its tech so the device geometry + spacing track the requested PDK through
    # every engine.
    from clay.layout.annotate_to_solver import _resolve_tech
    _tech = _resolve_tech(tech if tech is not None else pdk)
    if native_geometry is not None and (engine != "milp" or style_reference is not None):
        raise ValueError("native geometry requires the exact generic placement consumer")

    def _solve(g: int, rt: float):
        if engine == "seqpair":
            from clay.layout.seqpair_placement import (
                place_seqpair, relative_order_from_intent, solve_placement,
            )
            if style_reference is not None:
                from clay.layout.style_transfer import transfer_style
                from clay.layout.generic_placement import (
                    annotated_cell_from_cell_def, _build_devices_from_clay,
                )
                base = relative_order_from_intent(
                    cell_def, intent, source_cir=source_cir, tech=_tech
                )
                transferred = transfer_style(list(base.devices), style_reference).order
                annotated, _ = annotated_cell_from_cell_def(cell_def, source_cir)
                return solve_placement(
                    transferred, _build_devices_from_clay(annotated, _tech), tech=_tech
                )
            # Lock every device the CONVERGENCE CRITERIA tie into a matched /
            # common-centroid group (criteria carry device groups the parsed
            # placement constraints may not -- e.g. a 3-device cc cluster). The
            # seqpair wide-2D packer keeps these on their own axis-centred rows so
            # their group stays centroid-balanced (else packing would scatter the
            # cluster across a wide row and break matched_cc_balance).
            lock_devices: set = set()
            for grp in getattr(criteria, "common_centroid", ()) or ():
                lock_devices.update(grp)
            for pr in getattr(criteria, "matched_pairs", ()) or ():
                lock_devices.update(pr)
            for pr in getattr(criteria, "device_pairs", ()) or ():
                lock_devices.update(pr)
            return place_seqpair(
                cell_def, intent, source_cir=source_cir, tech=_tech,
                lock_devices=lock_devices,
            )
        extra = {} if native_composite_row is None else {'native_composite_row': native_composite_row}
        return place_generic(
            cell_def, intent, source_cir=source_cir, grid_nm=g, rel_tol=rt,
            tech=_tech, native_geometry=native_geometry, **extra,
        )

    placement = _solve(grid_nm, rel_tol)

    history: List[PlacementIteration] = []
    converged = False
    residual = 0.0
    unmet = ""

    for i in range(max_iters):
        intent_res = _check_intent(criteria, placement)
        if drc is not None:
            drc_res = drc(placement)  # caller-supplied predicate / result
            if isinstance(drc_res, bool):
                drc_res = _DrcResult(
                    drc_res, None,
                    int(required_inter_device_spacing_nm(
                        pdk_layers or Gf180mcuLayerMap())),
                    0, "caller-supplied drc predicate",
                )
        else:
            drc_res = _placement_drc_legal(placement, pdk_layers)

        intent_ok = bool(getattr(intent_res, "satisfied", False))
        kind, resid, action = _worst_obligation(intent_res, drc_res)

        history.append(
            PlacementIteration(
                index=i,
                symmetry=float(placement.symmetry_score),
                intent_satisfied=intent_ok,
                drc_legal=drc_res.legal,
                residual=float(resid),
                worst_obligation=(
                    kind or "none"
                ),
                action=action,
                grid_nm=grid_nm,
                rel_tol=rel_tol,
                min_gap_nm=drc_res.min_gap_nm,
                required_gap_nm=drc_res.required_nm,
            )
        )

        if intent_ok and drc_res.legal:
            converged = True
            residual = 0.0
            unmet = ""
            break

        # not converged: record the residual + worst obligation, then adjust
        # ONE generic knob toward it and re-solve.
        residual = float(resid)
        unmet = kind

        if action == "report_gap":
            # the worst unmet obligation (common-centroid / compactness) has no
            # lever in this placement engine -- the MILP places mirror pairs but
            # does not express a coincident-centroid placement constraint. Report
            # it as an engine-capability gap (do NOT burn iterations on an
            # ineffective knob, do NOT loop forever).
            prefix, _ = _worst_unmet_intent_obligation(intent_res)
            unmet = (
                f"{kind}:{prefix} (no placement lever in this engine for a "
                f"{prefix} obligation; needs a common-centroid MILP constraint)"
            )
            break

        adjusted = False
        if action == "widen_spacing":
            new_grid = min(_GRID_NM_MAX, grid_nm + _GRID_NM_STEP)
            if new_grid != grid_nm:
                grid_nm = new_grid
                adjusted = True
        elif action == "tighten_symmetry":
            new_tol = max(_REL_TOL_MIN, rel_tol * _REL_TOL_FACTOR)
            if new_tol != rel_tol:
                rel_tol = new_tol
                adjusted = True

        if not adjusted:
            # the relevant knob is saturated: the obligation is unsatisfiable
            # within this engine's lever range. Report it (do NOT loop forever).
            unmet = (
                f"{kind} (knob saturated: grid_nm={grid_nm}, "
                f"rel_tol={rel_tol:g})"
            )
            break

        prev_sig = (
            tuple(sorted((str(k), tuple(v.bbox)) for k, v in placement.devices.items())),
            float(placement.symmetry_score),
        )
        placement = _solve(grid_nm, rel_tol)
        next_sig = (
            tuple(sorted((str(k), tuple(v.bbox)) for k, v in placement.devices.items())),
            float(placement.symmetry_score),
        )
        if next_sig == prev_sig:
            # The adjusted knob produced the exact identical device placement
            # (e.g. an atomic row-program compaction that is invariant to grid_nm
            # / rel_tol). Stop immediately instead of repeating identical solves.
            unmet = (
                f"{kind} (knob invariant: grid_nm={grid_nm}, "
                f"rel_tol={rel_tol:g})"
            )
            break

    iters = len(history)

    # v3: record this run's outcome so a later matching context can seed from it
    # (default OFF -- only when a memory + learning + fingerprint are present).
    if memory is not None and enable_learning and _fp is not None:
        memory.record(
            _fp,
            {"grid_nm": float(grid_nm), "rel_tol": float(rel_tol)},
            converged=converged,
            iters=iters,
            residual=float(residual if not converged else 0.0),
        )

    note = (
        f"converged in {iters} iter(s)"
        if converged
        else f"unmet={unmet!r} residual={residual:.4f} after {iters} iter(s)"
    )
    return ConvergedPlacement(
        placement=placement,
        iters=iters,
        converged=converged,
        history=tuple(history),
        residual=residual if not converged else 0.0,
        unmet="" if converged else (unmet or "unknown"),
        note=note,
    )


__all__ = [
    "PlacementIteration",
    "ConvergedPlacement",
    "place_to_intent",
]
