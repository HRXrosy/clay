"""P6 dual-model reconcile: native union-find vs independent KLayout extraction.

Two models look at the ONE canonical as-built layout and must agree:

* the router's FAST native model -- union-find over :mod:`clay.layout.connectivity`,
  surfaced as the ``route_amp_leakfree`` metrics (``n_connected``/``n_required_nets``,
  ``n_shorts``, ``geometric_open_nets``);
* the INDEPENDENT KLayout extraction gate -- :func:`clay.layout.route.lvs_gate.lvs_gate`,
  a different implementation (foundry-style ``LayoutToNetlist`` + device-vs-schematic
  compare) so the two do not share blind spots.

When the native model declares the layout clean but KLayout reports an unmatched
netlist, that is a native BLIND SPOT -- the precise failure mode that let cap-plate
shorts and floating caps hide for several rounds during LVS bring-up. :func:`reconcile`
makes that disagreement explicit so a sign-off gate can fail on it automatically.

Both models MUST read the SAME canonical bundle: pass the dict returned by
``route_amp_leakfree(...)`` (it carries ``_bundle``). Reconcile NEVER re-routes -- doing
so would re-introduce the path drift this module exists to eliminate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = ["ReconcileVerdict", "reconcile"]


@dataclass
class ReconcileVerdict:
    """The outcome of comparing the two independent sign-off models.

    The KLayout side is the FULL sign-off predicate -- LVS-match AND real-deck
    DRC-clean (``klayout_signoff_clean``), NOT LVS alone. ``klayout_matched`` /
    ``klayout_drc_clean`` are kept as the two components so a caller can see WHICH
    half disagrees with native. ``agree`` compares native's clean verdict against
    ``klayout_signoff_clean`` -- so a layout the native model calls clean while the
    foundry deck still fires DRC is a native BLIND SPOT (native does not model the
    deck), exactly the apples-to-oranges the LVS-only comparison hid.
    """

    native_clean: bool
    klayout_matched: bool
    klayout_ran: bool
    agree: bool
    native_summary: str
    klayout_unmatched: Dict[str, list] = field(default_factory=lambda: {"nets": [], "devices": []})
    detail: str = ""
    klayout_drc_clean: bool = False
    klayout_drc_ran: bool = False
    klayout_drc_total: int = 0
    klayout_drc_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def klayout_signoff_clean(self) -> bool:
        """The FULL KLayout sign-off predicate: LVS matched AND real-deck DRC clean."""
        return bool(self.klayout_matched and self.klayout_drc_clean)


def _native_clean(metrics: Dict[str, Any]) -> tuple[bool, str]:
    """Decide whether the router's fast native model considers the layout clean, and
    summarise the evidence. Clean := every required net connected, no shorts, no
    geometric opens, route succeeded."""
    n_conn = int(metrics.get("n_connected", 0))
    n_req = int(metrics.get("n_required_nets", 0))
    n_shorts = int(metrics.get("n_shorts", 0))
    opens = list(metrics.get("geometric_open_nets", []) or [])
    success = bool(metrics.get("route_success", False))
    clean = success and n_shorts == 0 and not opens and n_req > 0 and n_conn == n_req
    summary = (
        f"native: connected={n_conn}/{n_req} shorts={n_shorts} "
        f"opens={len(opens)} route_success={success} -> clean={clean}"
    )
    if opens:
        summary += f"  open_nets={opens}"
    return clean, summary


def reconcile(
    routed_or_bundle: Any,
    intent: Any,
    *,
    tech_id: str,
    klayout_bin: Optional[Any] = None,
    timeout_sec: float = 300.0,
    drc_verdict: Optional[Any] = None,
) -> ReconcileVerdict:
    """Run BOTH sign-off models on the one canonical as-built layout and report
    whether they agree.

    ``routed_or_bundle`` should be the dict from :func:`route_amp_leakfree` (carrying the
    canonical ``_bundle`` + native metrics). The KLayout gate consumes that exact bundle,
    so both models see identical geometry -- any disagreement is a genuine model gap, not
    a path-drift artefact.

    The KLayout side is the FULL sign-off predicate -- LVS-match AND real foundry-deck
    DRC-clean -- so a layout the native model declares clean while the deck still fires
    DRC is caught as a native BLIND SPOT (the native model does not run the deck). Pass a
    precomputed :class:`clay.layout.route.drc_gate.DrcVerdict` as ``drc_verdict`` to avoid
    a second deck run when the caller already has one (the obligation set does); otherwise
    reconcile runs the deck itself.
    """
    from clay.layout.route.lvs_gate import lvs_gate

    if isinstance(routed_or_bundle, dict) and "_bundle" in routed_or_bundle:
        metrics = routed_or_bundle
    else:
        raise ValueError(
            "reconcile requires the canonical production result (a dict with '_bundle'); "
            "pass route_amp_leakfree(...) directly so both models read the SAME layout."
        )

    native_clean, native_summary = _native_clean(metrics)

    verdict = lvs_gate(
        routed_or_bundle, intent, tech_id=tech_id,
        klayout_bin=klayout_bin, timeout_sec=timeout_sec,
    )
    klayout_matched = bool(verdict.matched)
    klayout_ran = bool(verdict.ran)

    # the DRC half of the sign-off predicate (LVS-match AND DRC-clean). Reuse a
    # caller-supplied verdict (the obligation set computes one) or run the deck here.
    if drc_verdict is None:
        from clay.layout.route.drc_gate import drc_gate
        drc_verdict = drc_gate(routed_or_bundle, intent, tech_id=tech_id,
                               klayout_bin=klayout_bin)
    drc_ran = bool(getattr(drc_verdict, "ran", False))
    drc_clean = bool(getattr(drc_verdict, "clean", False))
    drc_total = int(getattr(drc_verdict, "total", 0))
    drc_counts = dict(getattr(drc_verdict, "counts_by_rule", {}) or {})

    klayout_signoff_clean = bool(klayout_matched and drc_clean)
    agree = native_clean == klayout_signoff_clean
    detail_lines = [
        native_summary,
        f"klayout: ran={klayout_ran} matched={klayout_matched} "
        f"unmatched_nets={len(verdict.unmatched['nets'])} "
        f"unmatched_devices={len(verdict.unmatched['devices'])}",
        f"drc: ran={drc_ran} clean={drc_clean} total={drc_total} counts={drc_counts}",
        f"klayout_signoff_clean(matched AND drc_clean)={klayout_signoff_clean}",
        f"AGREE={agree}",
    ]
    if not agree:
        direction = (
            "native-CLEAN but KLayout-NOT-SIGNOFF-CLEAN (native blind spot: "
            "LVS-unmatched and/or real-deck DRC violations the native model misses)"
            if native_clean
            else "native-DIRTY but KLayout-SIGNOFF-CLEAN (native over-strict)"
        )
        detail_lines.append(f"DISAGREEMENT: {direction}")
        for r in verdict.unmatched["devices"]:
            detail_lines.append(f"   dev: {r}")
        for r in verdict.unmatched["nets"]:
            detail_lines.append(f"   net: {r}")
        for rule, n in sorted(drc_counts.items()):
            detail_lines.append(f"   drc: {rule} x{n}")

    return ReconcileVerdict(
        native_clean=native_clean,
        klayout_matched=klayout_matched,
        klayout_ran=klayout_ran,
        agree=agree,
        native_summary=native_summary,
        klayout_unmatched={
            "nets": list(verdict.unmatched["nets"]),
            "devices": list(verdict.unmatched["devices"]),
        },
        detail="\n".join(detail_lines),
        klayout_drc_clean=drc_clean,
        klayout_drc_ran=drc_ran,
        klayout_drc_total=drc_total,
        klayout_drc_counts=drc_counts,
    )
