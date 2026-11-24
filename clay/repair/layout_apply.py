"""Close the layout-intent repair loop: APPLY a proposed fix, re-extract, re-check.

The layout-intent contract today DECLARES the net-level parasitic intent
(``clay.layout.intent``), CHECKS it against a real PEX extraction
(``clay.layout.intent_obligations``), and PROPOSES a routing move for a violated
obligation (``clay.repair._layout_ops.propose_layout_repair``). What was missing
is the last edge of the loop: actually APPLYING the proposed move to the layout,
re-extracting parasitics, and re-checking the obligation -- a real closed loop on
the comparator with real Magic PEX.

This module supplies that edge for the comparator's ``clk_regen`` obligation. The
applicable knob is the grounded clk shield strip: the committed router always
draws it (``build_routed_comparator(shield=True)``, the DRC-clean / 6-of-6-pass
layout); omitting it (``shield=False``) is a degraded layout where clk couples
unequally into the regen pair, so the ``clk_regen`` obligation fails. The
``insert_gnd_shield`` operator the proposer emits for that obligation maps onto
exactly this knob.

The driver reuses the ``RepairLoop``'s ``apply_and_reverify`` shape: the repair
move is realized by an ``apply_and_reverify`` callable that rebuilds the layout
with the knob set, re-runs Magic PEX, and re-checks the intent. Like the rest of
``clay.repair``, closing the loop is a repair-axis fact -- it never certifies a
verification PASS; the verifier remains the only authority on that.

The actual extraction goes through ``clay.layout.pex.MagicPexBackend`` (real
Magic ext2spice in Docker), so callers must gate on ``magic_runner.magic_available()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from clay.layout.intent import LayoutIntent, LayoutObligation
from clay.layout.intent_obligations import check_layout_intent

from ._layout_ops import KIND_CLK_REGEN, propose_layout_repair

# The committed comparator GDS top-cell name (kept identical to the rest of the
# layout-intent tests so the same Magic flow extracts it).
_CELL_NAME = "routed_comparator"

# Map an obligation kind -> the layout knob that realizes its repair, plus the
# value that ENABLES the fix. The clk_regen obligation is closed by drawing the
# grounded clk shield strip, i.e. setting build_routed_comparator(shield=True).
# (The proposer emits insert_gnd_shield for clk_regen; this is the knob it names.)
_KIND_KNOB = {KIND_CLK_REGEN: ("shield", True)}


def extract_and_check(
    intent: LayoutIntent,
    *,
    shield: bool = True,
    tmp_dir,
    gds_name: str = "layout_apply.gds",
) -> List[LayoutObligation]:
    """Build the comparator at the given knob, emit GDS, run REAL Magic PEX, check.

    Rebuilds the routed comparator with ``build_routed_comparator(shield=shield)``,
    emits it to ``tmp_dir/gds_name``, extracts parasitics with the real Magic
    backend, and returns the obligations from ``check_layout_intent``. This is the
    full DECLARE -> (build) -> extract -> CHECK leg; one real extraction per call.
    """
    # Imported lazily so importing this module never pulls in klayout / Magic.
    from clay.layout.comparator_router import (
        build_routed_comparator,
        emit_routed_comparator,
    )
    from clay.layout.pex import MagicPexBackend

    tmp_dir = Path(tmp_dir)
    gds = tmp_dir / gds_name
    cell = build_routed_comparator(shield=shield)
    emit_routed_comparator(cell, str(gds), cell_name=_CELL_NAME)
    report = MagicPexBackend().extract(str(gds), _CELL_NAME, "gf180mcu")
    return check_layout_intent(intent, report)


def _obligation_of_kind(
    obligations: List[LayoutObligation], kind: str
) -> Optional[LayoutObligation]:
    for o in obligations:
        if o.kind == kind:
            return o
    return None


def close_layout_repair_loop(
    intent: LayoutIntent,
    tmp_dir,
    *,
    kind: str = KIND_CLK_REGEN,
) -> Dict[str, object]:
    """Drive the layout-intent repair APPLY loop end-to-end on a real extraction.

    The closed loop, reusing the ``RepairLoop`` apply_and_reverify shape:

      1. START DEGRADED: extract+check with the knob OFF (``shield=False``); the
         ``clk_regen`` obligation FAILS (clk couples unequally into the regen
         pair without the grounded shield strip).
      2. PROPOSE: ask ``propose_layout_repair`` on the failing obligation -> it
         returns an ``insert_gnd_shield`` (LAYOUT_AWARE) proposal naming the
         shield knob.
      3. APPLY + REVERIFY: an ``apply_and_reverify`` callable sets the named knob
         (``shield=True``), rebuilds the layout, re-runs Magic PEX, and re-checks
         the intent. The ``clk_regen`` obligation now PASSES.

    Returns a before/after record::

        {
          "kind": "clk_regen",
          "obligation": <pre-fix LayoutObligation>,   # the failing one
          "before_imbalance": float,                  # shield OFF
          "proposal": <dict from propose_layout_repair>,
          "knob": "shield",
          "after_imbalance": float,                   # shield ON
          "after_status": "pass" | "fail",
          "fixed": bool,                              # passes budget & improved
        }

    Closing the loop is a repair-axis fact; it does NOT certify a verification
    PASS. Two real extractions total (degraded, then repaired).
    """
    knob, fix_value = _KIND_KNOB.get(kind, ("shield", True))

    # --- 1. START DEGRADED: knob off -> the obligation should fail. ---------
    degraded = extract_and_check(
        intent, shield=False, tmp_dir=tmp_dir, gds_name="layout_apply_degraded.gds"
    )
    before = _obligation_of_kind(degraded, kind)
    if before is None:
        raise ValueError(f"intent declares no {kind!r} obligation to repair")

    # --- 2. PROPOSE a routing move for the failing obligation. --------------
    proposal = propose_layout_repair(before)
    if not isinstance(proposal, dict):
        # A passing obligation (None) or an unknown kind (RepairStatus) means the
        # degraded layout did not expose the defect the loop is meant to close.
        raise ValueError(
            f"expected a repair proposal for a failing {kind!r} obligation, "
            f"got {proposal!r} (before_imbalance={before.measured_imbalance:.4f}, "
            f"rel_tol={before.rel_tol:.4f})"
        )

    # --- 3. APPLY + REVERIFY: rebuild with the knob set, re-PEX, re-check. ---
    # This is the RepairLoop apply_and_reverify shape: a callable that takes the
    # proposed move, applies the named knob, and returns a fresh obligation set.
    def apply_and_reverify(prop: dict) -> List[LayoutObligation]:
        # The proposal names the knob to turn (prop["knob"]); applying the fix
        # sets it to its enabling value and re-extracts.
        applied_knob = prop.get("knob", knob)
        kwargs = {applied_knob: fix_value} if applied_knob == "shield" else {"shield": fix_value}
        return extract_and_check(
            intent, **kwargs, tmp_dir=tmp_dir, gds_name="layout_apply_repaired.gds"
        )

    repaired = apply_and_reverify(proposal)
    after = _obligation_of_kind(repaired, kind)
    if after is None:  # pragma: no cover - intent is fixed for the comparator
        raise ValueError(f"re-check lost the {kind!r} obligation")

    fixed = after.status == "pass" and after.measured_imbalance < before.measured_imbalance
    return {
        "kind": kind,
        "obligation": before,
        "before_imbalance": before.measured_imbalance,
        "proposal": proposal,
        "knob": proposal.get("knob", knob),
        "after_imbalance": after.measured_imbalance,
        "after_status": after.status,
        "fixed": bool(fixed),
    }


__all__ = ["extract_and_check", "close_layout_repair_loop"]
