"""Phase R -- the ROUTING convergence entry for the Clay intent-convergent
place + route engine.

Directive (design doc, 2026-05-29): placement converges first (Phase P,
:mod:`clay.layout.place_refine`); ONLY THEN route. Production routing runs the
GENERAL negotiated-congestion PathFinder engine
(:class:`~clay.layout.route.engine_pathfinder.PathFinderEngine`) via
:func:`clay.layout.route.route_cell.route_cell`, wrapped here by
:func:`route_cell_to_intent`.

route_cell is DETERMINISTIC and produces a net-complete, cross-net-short-free
route in a single pass (verified by its own union-find oracle), so it needs no
spacing-knob convergence loop. Acceptance requires both native connectivity and
the selected engine's exact success verdict; a Gridless deck-spacing residual
must not be promoted as converged merely because every pin is connected. NO
per-circuit branch -- the device net binding
comes from ``cell_def.devices`` when materialised, else from the bound-``.cir``
``device_records`` (the SAME netlist the placer + router consume); diff-pair
mirroring rides the declared ``intent``.

(The legacy hand-rolled ``route_generic`` + its spacing-knob ``route_to_intent``
convergence loop are RETIRED from production -- they self-converged a far weaker
greedy router that route_cell/PathFinder replaces. A snapshot of the retired
loop lives in ``trash/route_refine_route_to_intent_legacy.py``.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from clay.layout.place_blocks_2d import FinalPlacementCertificate


# ── result + history ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingIteration:
    """One iteration of the routing convergence (audit trail)."""

    index: int
    shorts: Optional[int]
    intent_satisfied: bool
    drc_clean: bool
    lvs_match: bool
    residual: float
    worst_obligation: str
    action: str
    spacing_extra_nm: int


# ── the GENERAL route_cell production entry (PathFinder engine) ────────────────


@dataclass
class ConvergedRouteCell:
    """The Phase-R result for the GENERAL ``route_cell`` (PathFinder) production
    path.

    ``cell`` is the generic-``RoutedCell``-shaped adapter
    (``RouteCellResult.to_routed_cell()``) the production flow + the
    fault-intent / DRC-diagnosis consumers read. ``result`` is the underlying
    :class:`~clay.layout.route.route_cell.RouteCellResult` (carries ``routed`` /
    ``geometry`` / ``placement_um`` / ``pins_by_net`` / ``tech`` for the full-stack
    emit). The remaining fields record the flow's convergence provenance
    (converged / iters / residual / unmet / note).
    """

    cell: Any
    result: Any
    iters: int
    converged: bool
    history: Tuple[RoutingIteration, ...]
    residual: float = 0.0
    unmet: str = ""
    note: str = ""

    @property
    def shorts(self) -> Optional[int]:
        sc = getattr(self.result, "shorts", None)
        if sc is None:
            return None
        return len(getattr(sc, "merged_pairs", ()) or ())


def route_cell_to_intent(
    placement,
    cell_def,
    intent,
    pdk=None,
    *,
    device_records: Optional[Dict[str, Dict]] = None,
    max_iters: int = 1,
    engine: Optional[Any] = None,
    placement_override: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    final_placement_certificate: Optional[FinalPlacementCertificate] = None,
    member_ring_pad_top_nm: Tuple[Tuple[str, int], ...] = (),
    member_ring_pad_bottom_nm: Tuple[Tuple[str, int], ...] = (),
    shared_well_guard_envelope_plans: tuple = (),
    shared_well_guard_ring_inventory: Any = None,
    execution_scope: Optional[Any] = None,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
    group_fallback_evidence_dir: Any = None,
    diagnostic_observer: Any = None,
    pre_route_gate: Any = None,
    diagnostic_compact: Any = None,
    diagnostic_policy: Any = None,
    diagnostic_source_ticket: Any = None,
) -> ConvergedRouteCell:
    """Route a placed cell through the GENERAL negotiated-congestion
    :class:`~clay.layout.route.engine_pathfinder.PathFinderEngine` (via
    :func:`clay.layout.route.route_cell.route_cell`) and return a converged result.

    This is the production routing entry for the general router. route_cell is
    DETERMINISTIC and produces a net-complete, cross-net-short-free route (verified
    by its own union-find oracle) by construction in a single pass, so the
    acceptance here requires ``RouteCellResult.net_complete`` (every required net
    present + connected, no cross-net union-find short) AND the selected engine's
    exact success verdict. NO per-circuit branch: the
    device net binding comes from ``cell_def.devices`` when materialised, else from
    the bound-``.cir`` ``device_records`` (the SAME netlist the placer + router
    consume); diff-pair mirroring rides the declared ``intent`` -- driven only by
    intent + PDK.

    ``device_records`` is the parsed-``.cir`` ``{name -> {kind, W/L/nf | value,
    pins}}`` map (``parse_cir_mos_records`` + ``parse_cir_passive_records``); pass it
    when the parsed cell's devices are topology-only (their net bindings live in the
    netlist). ``max_iters`` is accepted for signature parity with the historical
    convergence entry (route_cell needs no spacing-knob convergence loop).

    ``engine`` (default None -> byte-identical PathFinder) forwards an ALTERNATE
    :class:`~clay.layout.route.engine.RouteEngine` strategy into route_cell --
    the seam the resolved ``@route_policy(engine=)`` declaration reaches through
    (``clay.layout.route.engine_registry.route_engine_from_policy``).

    ``placement_override`` is the exact, ring-free route placement produced by
    the canonical declared-policy placement stage.  It is forwarded unchanged;
    :func:`route_cell` remains the sole owner of ring materialization.

    ``final_placement_certificate`` is the typed final-master certificate paired
    with that override.  It is forwarded unchanged so :func:`route_cell` can
    replay it only after installing the override and observing its exact geometry.
    """
    from clay.core_contract import VerificationStatus
    from clay.layout.route.route_cell import (
        BringupDiagnosticRouteResult,
        RouteExecutionScope,
        RoutePlanningStop,
        route_cell,
    )

    if (
        execution_scope is RouteExecutionScope.BRINGUP_DIAGNOSTIC
        and not bool(getattr(
            engine, "supports_deterministic_negotiation", False
        ))
    ):
        engine_type = (
            "default:PathFinderEngine"
            if engine is None
            else f"{type(engine).__module__}.{type(engine).__qualname__}"
        )
        raise RoutePlanningStop(
            phase="DIAGNOSTIC_CAPABILITY",
            state="DETERMINISTIC_NEGOTIATION_ENGINE_REQUIRED",
            evidence=(
                "BRINGUP_DIAGNOSTIC requires an engine with deterministic "
                "negotiated-congestion capability",
                engine_type,
            ),
            status=VerificationStatus.UNSUPPORTED,
        )

    effective_scope = (
        RouteExecutionScope.SIGNOFF
        if execution_scope is None
        else execution_scope
    )
    res = route_cell(
        placement,
        cell_def,
        intent,
        pdk,
        device_records=device_records,
        engine=engine,
        placement_override=placement_override,
        final_placement_certificate=final_placement_certificate,
        member_ring_pad_top_nm=member_ring_pad_top_nm,
        member_ring_pad_bottom_nm=member_ring_pad_bottom_nm,
        shared_well_guard_envelope_plans=shared_well_guard_envelope_plans,
        shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
        execution_scope=effective_scope,
        intent_fact_set=intent_fact_set,
        source_inventory=source_inventory,
        **({"diagnostic_observer": diagnostic_observer, "diagnostic_compact": diagnostic_compact,
            "diagnostic_policy": diagnostic_policy} if diagnostic_observer is not None else {}),
        **({"pre_route_gate": pre_route_gate, "diagnostic_compact": diagnostic_compact,
            "diagnostic_policy": diagnostic_policy} if pre_route_gate is not None else {}),
        **({"diagnostic_source_ticket": diagnostic_source_ticket} if diagnostic_source_ticket is not None else {}),
        **({"group_fallback_evidence_dir": group_fallback_evidence_dir}
           if group_fallback_evidence_dir is not None else {}),
    )
    routed = res.to_routed_cell()
    sc = getattr(res, "shorts", None)
    n_short = len(getattr(sc, "merged_pairs", ()) or [])
    diagnostic_only = type(res) is BringupDiagnosticRouteResult
    if diagnostic_only:
        # Keep the legacy generic adapter available to read its geometry, but
        # make its non-signoff lineage explicit and remove its completeness
        # surface.  A downstream consumer must never mistake this feasibility
        # observation for an ordinary converged RoutedCell.
        routed.diagnostic_only = True
        routed.signoff_eligible = False
        routed.diagnostic_summary_digest = res.diagnostic_summary_digest
        routed.net_complete = None
        converged = False
        residual = 1.0
        unmet = (
            "NON_SIGNOFF_BRINGUP_DIAGNOSTIC: feasibility geometry cannot "
            "establish routing convergence, intent satisfaction, or DRC closure"
        )
        worst_obligation = "diagnostic_only"
        action = "route_cell:bringup_diagnostic"
    else:
        engine_result = getattr(res, "routed", None)
        engine_success = bool(getattr(engine_result, "success", True))
        converged = bool(res.net_complete and engine_success)
        residual = (
            0.0
            if converged
            else (n_short / (n_short + 1.0) if n_short else 1.0)
        )
        if converged:
            unmet = ""
            worst_obligation = "none"
        elif not res.net_complete:
            unmet = (
                f"route_cell not net-complete: {res.n_connected}/{res.n_required} "
                f"nets connected, unconnected={res.unconnected_nets}, "
                f"cross-net shorts={res.merged_pairs}"
            )
            worst_obligation = "net_complete"
        else:
            deck_pairs = list(getattr(
                engine_result, "deck_residual_pairs", ()
            ) or ())
            route_stops = list(getattr(
                engine_result, "route_stop_refs", ()
            ) or ())
            unsatisfied_constraints = sorted(
                repr(constraint)
                for constraint in (
                    getattr(engine_result, "unsatisfied", ()) or ()
                )
            )
            detail_parts = [
                f"deck_residual_pairs={deck_pairs}",
                f"route_stop_refs={route_stops}",
                f"unsatisfied_constraints={unsatisfied_constraints}",
            ]
            unconnected_nets = sorted(
                net
                for net, net_route in (
                    getattr(engine_result, "nets", {}) or {}
                ).items()
                if not getattr(net_route, "connected", True)
            )
            shorts_list = list(getattr(engine_result, "shorts", ()) or ())
            overused_cells = list(
                getattr(engine_result, "best_overused_cells", ()) or ()
            )
            pad_contended = list(
                getattr(engine_result, "best_pad_contended_cells", ()) or ()
            )
            negotiation_rounds = getattr(
                engine_result, "negotiation_rounds", None
            )
            failure_reasons = list(
                getattr(engine_result, "convergence_failure_reasons", ()) or ()
            )
            if negotiation_rounds is not None:
                detail_parts.append(f"negotiation_rounds={negotiation_rounds}")
            if unconnected_nets:
                detail_parts.append(f"unconnected_nets={unconnected_nets}")
            if shorts_list:
                detail_parts.append(f"shorts={shorts_list[:8]}")
            if overused_cells:
                detail_parts.append(
                    f"overused_cells={overused_cells[:8]} (total={len(overused_cells)})"
                )
            if pad_contended:
                detail_parts.append(
                    f"pad_contended={pad_contended[:8]} (total={len(pad_contended)})"
                )
            if failure_reasons:
                detail_parts.append(f"reasons={failure_reasons}")
            unmet = "route engine did not converge: " + ", ".join(detail_parts)
            worst_obligation = "engine_success"
        action = "route_cell"
    history = (
        RoutingIteration(
            index=0,
            shorts=n_short,
            intent_satisfied=converged,
            drc_clean=converged,
            lvs_match=False,
            residual=float(residual),
            worst_obligation=worst_obligation,
            action=action,
            spacing_extra_nm=0,
        ),
    )
    note = (
        "route_cell net-complete (general PathFinder engine)"
        if converged
        else unmet
    )
    return ConvergedRouteCell(
        cell=routed,
        result=res,
        iters=1,
        converged=converged,
        history=history,
        residual=0.0 if converged else float(residual),
        unmet=unmet,
        note=note,
    )


__all__ = [
    "RoutingIteration",
    "ConvergedRouteCell",
    "route_cell_to_intent",
]
