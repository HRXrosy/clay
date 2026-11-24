"""UNIFIED layout signoff: one verdict certifying BOTH parasitics AND placement.

The layout-intent contract is checked along two independent axes today, each with
its own certify path:

  * NET-LEVEL PARASITICS -- ``clay.layout.intent_certify.certify_layout_intent``
    measures each declared :class:`~clay.layout.intent.ParasiticBudget` against a
    real :class:`~clay.layout.pex.base.PexReport` and signs a token iff every
    parasitic budget is within its RELATIVE tolerance.
  * DEVICE PLACEMENT -- ``clay.layout.placement_symmetry`` measures how
    mirror-symmetric each matched device pair's centroids are about the cell's
    vertical axis, straight off the placed geometry (no Magic / PEX).

A cell that routes its parasitics perfectly but places a differential device pair
asymmetrically (or vice-versa) is not actually signed off, yet the two checks
hand back two separate verdicts. This module fuses them into ONE certify + score
+ proof token: it maps the placement obligations into the SAME proof-obligation
row shape the parasitic obligations already use, feeds the COMBINED required rows
to the single proof-token gate (``clay.verifier._certify_obligation_rows``), and
so the token is signed only when BOTH axes pass. A placement failure withholds
the token exactly the way a parasitic failure does -- the token is NEVER
fabricated.

Everything reads a ``PexReport`` for the parasitic side and the placed geometry
for the placement side, so the whole signoff is unit-testable WITHOUT Magic /
Docker by passing the real ``build_routed_comparator()`` (fast geometry) and a
synthetic ``PexReport``. All thresholds are RELATIVE fractions (project-wide
rule: never absolute aF / nm deltas).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from clay.signoff_obligation_trace import _row
from clay.verifier import _certify_obligation_rows

from .intent import LayoutIntent, derive_layout_intent
from .intent_obligations import check_layout_intent, check_shield_intent
from .intent_score import layout_intent_score
from .intent_signoff import layout_intent_obligation_rows
from .pex.base import PexReport
from .placement_symmetry import (
    mirror_placement_obligations,
    placement_symmetry_obligations,
)

# Source kind tag distinguishing placement-symmetry rows from the parasitic
# ``layout_intent_signoff`` rows while keeping the same row shape the proof-token
# gate consumes.
PLACEMENT_SOURCE_KIND = "layout_placement_signoff"
SHIELD_SOURCE_KIND = "layout_shield_signoff"
# The unified signoff carries no electrical PDK corner / temperature -- both axes
# are geometric / parasitic. Stamp the same neutral placeholders the parasitic
# certify path uses so the corner-aware token gate has consistent inputs.
_LAYOUT_PDK_CORNER = "layout"
_LAYOUT_TEMP = 27.0


def _placement_pair_token(pair) -> str:
    """``(a, b)`` -> ``a-b`` for a stable, unique obligation-id suffix."""
    a, b = pair
    return f"{a}-{b}"


def _placement_obligation_row(
    cell_name: str,
    obligation: Dict[str, object],
    *,
    required: bool,
) -> dict:
    """Map ONE placement-symmetry obligation dict into a proof-obligation row.

    The placement checker (``placement_symmetry_obligations``) returns plain dicts
    ``{pair, devices, dx_imbalance, dy_offset, rel_tol, status}``. This builds a
    row in the EXACT signoff shape the proof-token gate consumes -- reusing
    :func:`clay.signoff_obligation_trace._row` (the single source of truth for the
    dict shape) and then re-stamping the signoff defaults with a placement
    identity, the same way :mod:`clay.layout.intent_signoff` re-stamps the
    parasitic rows. So nothing about the row shape is duplicated here.

    The obligation id is ``placement_symmetry:<pairA>-<pairB>`` where the pair is
    the diff-pair NETS; because a single diff pair can be realised by several
    matched device pairs, the measured DEVICE labels are appended so each device
    pair gets a unique id (otherwise two device pairs of the same diff pair would
    collide on one id and the second would be dropped by the gate).
    """
    pair = tuple(obligation["pair"])              # the diff-pair NETS
    devices = tuple(obligation["devices"])        # the matched DEVICE labels
    status = str(obligation["status"])            # "pass" / "fail"
    check = _placement_pair_token(pair)
    dx = float(obligation["dx_imbalance"])
    dy = float(obligation["dy_offset"])
    rel_tol = float(obligation["rel_tol"])

    row = _row(
        cell_name=cell_name,
        obligation_kind="placement_symmetry",
        check=check,
        status=status,
        coverage={
            "artifact_kind": "placed_geometry",
            # placement is measured straight off the placed cell, no PEX tool run
            "ran_real_tool": True,
            "metric": "relative_centroid_mirror",
        },
        evidence={
            "dx_imbalance": dx,
            "dy_offset": dy,
            "worst": max(dx, dy),
            "rel_tol": rel_tol,
            "pair": list(pair),
            "devices": list(devices),
        },
    )
    # Re-stamp the signoff defaults with the placement identity. The shape (the
    # set of keys + nested source keys) is preserved; only the id / trace /
    # source_kind / required / source-detail values change. The id requested by
    # the contract is ``placement_symmetry:<pairA>-<pairB>``; the device labels
    # are folded into the trace id (and a uniqueness suffix on the obligation id
    # only when several device pairs share one diff pair) so no two rows collide.
    dev_token = "+".join(devices)
    match_kind = str(obligation.get("match_kind", "differential"))
    # Mirror-load placement obligations get a distinct id prefix so a caller can
    # tell a current-mirror placement blocker from a differential one at a glance
    # (and so the row's obligation_id NEVER collides with the differential pair
    # check for the same net pair).
    id_prefix = "mirror_placement" if match_kind == "mirror" else "placement_symmetry"
    obligation_id = f"{id_prefix}:{check}:{dev_token}"
    trace_id = f"tr:{cell_name}:{id_prefix}:{check}:{dev_token}"
    row["obligation_id"] = obligation_id
    row["trace_id"] = trace_id
    row["source_kind"] = PLACEMENT_SOURCE_KIND
    row["required"] = bool(required)
    row["source"] = {
        "cell": cell_name,
        "name": check,
        "field": id_prefix,
        "ir_symbol": f"{cell_name}.{id_prefix}.{check}.{dev_token}",
        "source_ref": {
            "kind": id_prefix,
            "path": f"cell[{cell_name}].{id_prefix}[{check}][{dev_token}]",
        },
    }
    # Surface match_kind on the evidence so the formatter / external consumers
    # can tell a differential pair obligation from a current-mirror one.
    row["evidence"]["match_kind"] = match_kind
    return row


def placement_symmetry_rows(
    cell_name: str,
    routed_cell,
    intent: LayoutIntent,
    *,
    required: bool = True,
    **placement_kw,
) -> List[dict]:
    """Proof-obligation rows for a cell's PLACEMENT-symmetry obligations.

    Runs :func:`placement_symmetry_obligations` and maps each resulting obligation
    dict into a row in the same shape ``clay.signoff_obligation_trace`` uses for
    DRC / LVS signoff. Each row's ``status`` is the obligation's own pass / fail
    verdict; ``required`` controls whether a failing placement blocks the proof
    token (default ``True``, matching how a parasitic budget blocks it).
    """
    obligations = placement_symmetry_obligations(routed_cell, intent, **placement_kw)
    return [
        _placement_obligation_row(cell_name, obligation, required=required)
        for obligation in obligations
    ]


def _shield_obligation_row(
    cell_name: str,
    shield_result: Dict[str, object],
    *,
    required: bool,
) -> dict:
    """Map ONE shield check result into a proof-obligation row.

    A declared shield is a geometric obligation (a grounded shape between the
    aggressor and the victim). The check is ``clay.layout.intent_obligations.
    check_shield_intent``, which returns plain dicts. We map each to a row in the
    SAME signoff shape the parasitic / placement axes use, so the proof-token
    gate treats a missing shield exactly like a DRC violation.
    """
    aggressor = str(shield_result.get("aggressor", "?"))
    victim = str(shield_result.get("victim", "?"))
    shield_net = str(shield_result.get("shield_net", "gnd"))
    shielded = bool(shield_result.get("shielded", False))
    geom = shield_result.get("geometry", {}) or {}
    reason = str(geom.get("reason", "")) if isinstance(geom, dict) else ""

    check = f"{aggressor}->{victim}"
    # Tri-state status: the bbox-heuristic geometric shield check returns
    # ``shielded=True`` (we found a grounded shape in the gap), or False with a
    # ``reason``. Two of those reasons mean the heuristic CANNOT MEASURE the
    # shield (the nets overlap so there is no gap to read; one of the two nets
    # is absent from the placed cell), which is honestly UNMEASURABLE -- the
    # heuristic was geometrically wrong-fit, not the layout was wrong. The third
    # reason ("no gnd shape in the gap") is a real geometric MISS -> fail. An
    # ``unmeasurable`` shield is non-pass (so it BLOCKS the token, matching the
    # parasitic-axis policy: a declared obligation that could not be honestly
    # measured is never silently a pass), but a future layer-aware shield check
    # can upgrade it to a real verdict without changing this contract.
    UNMEASURABLE_REASONS = {
        "aggressor and victim overlap (no gap)",
        "aggressor or victim net not present",
    }
    if shielded:
        status = "pass"
    elif reason in UNMEASURABLE_REASONS:
        status = "unmeasurable"
    else:
        status = "fail"
    row = _row(
        cell_name=cell_name,
        obligation_kind="shield",
        check=check,
        status=status,
        coverage={
            "artifact_kind": "placed_geometry",
            # geometric check: a routed cell IS a real artifact; no extraction
            # tool run, but the answer is read straight off the cell geometry.
            "ran_real_tool": True,
            "metric": "geometric_shield_between",
        },
        evidence={
            "aggressor": aggressor,
            "victim": victim,
            "shield_net": shield_net,
            "shielded": shielded,
            "reason": reason,
        },
    )
    obligation_id = f"shield:{aggressor}->{victim}"
    trace_id = f"tr:{cell_name}:shield:{aggressor}->{victim}"
    row["obligation_id"] = obligation_id
    row["trace_id"] = trace_id
    row["source_kind"] = SHIELD_SOURCE_KIND
    row["required"] = bool(required)
    row["source"] = {
        "cell": cell_name,
        "name": check,
        "field": "shield",
        "ir_symbol": f"{cell_name}.shield.{aggressor}_{victim}",
        "source_ref": {
            "kind": "shield",
            "path": f"cell[{cell_name}].shield[{aggressor}->{victim}]",
        },
    }
    return row


def _placement_pass_fraction(placement_obligations: List[Dict[str, object]]) -> float:
    """Fraction of placement obligations that pass (1.0 for the empty case).

    Mirrors the verdict convention: a cell that declared no diff pairs has no
    placement symmetry to violate, so an empty set is vacuously a full pass.
    """
    n = len(placement_obligations)
    if n == 0:
        return 1.0
    n_pass = sum(1 for o in placement_obligations if o.get("status") == "pass")
    return n_pass / n


def full_layout_signoff(
    cell,
    routed_cell,
    pex_report: PexReport,
    *,
    required: bool = True,
    axis: Optional[float] = None,
    intent: Optional[LayoutIntent] = None,
    check_shields: bool = True,
    **tol,
) -> Dict[str, object]:
    """Run the UNIFIED layout signoff over BOTH parasitics AND placement.

    Pipeline (one clean call):

      1. ``derive_layout_intent(cell, **tol)`` -- the declared layout intent
         (parasitic budgets + differential pairs).
      2. PARASITIC axis -- ``layout_intent_obligation_rows(intent, pex_report)``
         turns each parasitic budget into a proof-obligation row (via
         ``check_layout_intent``).
      3. PLACEMENT axis -- ``placement_symmetry_obligations(routed_cell, intent)``
         measures each matched device pair's centroid mirror-symmetry; each is
         mapped into a proof-obligation row in the SAME signoff shape.
      4. The COMBINED required rows (parasitic + placement) are fed to the SAME
         proof-token gate DRC / LVS use (``_certify_obligation_rows``): it returns
         ``(status, token)`` and signs a token only when EVERY required row is
         ``pass`` -- so the token is signed only when BOTH axes pass, and a
         placement failure withholds it exactly like a parasitic failure.

    ``cell`` is the cell NAME (str) routed through ``derive_layout_intent``; it may
    also be a cell object exposing ``.name``. ``routed_cell`` is the placed/routed
    cell whose geometry the placement check reads (e.g. ``build_routed_comparator()``).
    ``pex_report`` carries the parasitics for the budget check. ``axis`` is an
    optional explicit vertical symmetry axis for the placement check; ``**tol``
    forwards relative-tolerance overrides to ``derive_layout_intent``.

    Returns::

        {
          "cell": <cell name>,
          "status": "pass" | "fail" | ...,     # combined gate status
          "token": <signed proof token> | None,# None when EITHER axis withholds
          "passed": bool,                       # status == "pass"
          "blocking": [ <failing obligation_id>, ... ],  # parasitic AND/OR placement
          "rows": [ <combined proof-obligation row>, ... ],
          "parasitic_obligations": [ <LayoutObligation>, ... ],
          "placement_obligations": [ <placement obligation dict>, ... ],
          "score": { ... combined parasitic score + placement pass-fraction ... },
        }

    The token is ``None`` whenever any required obligation on EITHER axis fails;
    it is NEVER fabricated.
    """
    # Resolve cell name + intent. Explicit ``intent=`` wins; ``cell`` may be a
    # LayoutIntent (legal); otherwise we re-derive from the cell name (legacy).
    if intent is None and isinstance(cell, LayoutIntent):
        intent = cell
    cell_name = (
        intent.cell if intent is not None
        else (cell if isinstance(cell, str) else getattr(cell, "name", str(cell)))
    )
    if intent is None:
        intent = derive_layout_intent(cell_name, **tol)

    # ── Parasitic axis ───────────────────────────────────────────────────────
    parasitic_obligations = check_layout_intent(intent, pex_report)
    parasitic_rows = layout_intent_obligation_rows(
        intent, pex_report, required=required
    )

    # ── Placement axis (DIFF + MIRROR) ───────────────────────────────────────
    # v2: a five_t_ota / mirror-loaded cell declares ``mirror_pairs`` but no
    # ``matched_cc_balance`` budget for them (the two nodes are asymmetric by
    # design). Their PLACEMENT must still be mirror-symmetric -- the mirror
    # devices' centroids about the cell axis. ``placement_symmetry_obligations``
    # only checks the differential pairs; ``mirror_placement_obligations``
    # covers the current-mirror load pairs. The full signoff now merges both so
    # a mirror obligation also blocks the token.
    placement_kw: Dict[str, object] = {}
    if axis is not None:
        placement_kw["axis"] = axis
    diff_obls = (
        placement_symmetry_obligations(routed_cell, intent, **placement_kw)
        if intent.diff_pairs else []
    )
    mirror_obls = (
        mirror_placement_obligations(routed_cell, intent, **placement_kw)
        if intent.mirror_pairs else []
    )
    placement_obligations = list(diff_obls) + list(mirror_obls)
    placement_rows = [
        _placement_obligation_row(cell_name, ob, required=required)
        for ob in placement_obligations
    ]

    # ── Shield axis (geometric grounded-shape between aggressor/victim) ──────
    # A shield declared in the intent but not realised in the placed cell is a
    # silent regression: the proof-token gate must reject it the same way it
    # rejects a missing DRC clean run. We check shields only when the routed
    # cell exposes the geometric data; ``check_shield_intent`` returns each
    # shield's verdict as a plain dict. Suppressed via ``check_shields=False``
    # for callers that have no routed geometry to read (e.g. a parasitic-only
    # CI gate).
    shield_results: List[Dict[str, object]] = []
    shield_rows: List[dict] = []
    if check_shields and intent.shields:
        try:
            shield_results = check_shield_intent(intent, routed_cell)
        except (AttributeError, TypeError):
            # routed_cell does not expose .tagged geometry -- declared shields
            # remain unverified, surfaced below as unmeasurable rows so the
            # token is withheld rather than silently signed.
            shield_results = [
                {
                    "aggressor": s.aggressor,
                    "victim": s.victim,
                    "shield_net": s.shield_net,
                    "shielded": False,
                    "geometry": {"reason": "routed_cell has no .tagged geometry"},
                }
                for s in intent.shields
            ]
        shield_rows = [
            _shield_obligation_row(cell_name, r, required=required)
            for r in shield_results
        ]

    # ── Combined certify: ONE token over ALL axes ────────────────────────────
    rows = parasitic_rows + placement_rows + shield_rows
    status, token = _certify_obligation_rows(
        cell_name=cell_name,
        pdk_corner=_LAYOUT_PDK_CORNER,
        temp=_LAYOUT_TEMP,
        rows=rows,
    )

    # Blocking = every required row that is not a pass, named by its obligation
    # id so the caller can tell a parasitic blocker (``layout_intent:...``) from a
    # placement blocker (``placement_symmetry:...`` / ``mirror_placement:...``)
    # from a shield blocker (``shield:...``) at a glance.
    blocking: List[str] = [
        row["obligation_id"]
        for row in rows
        if row.get("required", True) and row["status"] != "pass"
    ]

    # ── Combined score: parasitic quality + placement pass-fraction ──────────
    parasitic_score = layout_intent_score(parasitic_obligations)
    placement_pass_fraction = _placement_pass_fraction(placement_obligations)
    # The combined scalar is the mean of the two axes' [0,1] quality numbers
    # (parasitic mean-quality + placement pass-fraction), so improving EITHER axis
    # raises it and a failing placement drags it down even when parasitics are
    # perfect. Both inputs are relative / scale-invariant, so the combined number
    # is too. The per-axis numbers are reported alongside so neither is lost.
    parasitic_quality = float(parasitic_score["score"])
    combined = (parasitic_quality + placement_pass_fraction) / 2.0
    n_shield_pass = sum(1 for r in shield_results if r.get("shielded"))
    score = {
        "combined": combined,
        "parasitic_score": parasitic_quality,
        "placement_pass_fraction": placement_pass_fraction,
        "parasitic": parasitic_score,
        "n_placement_obligations": len(placement_obligations),
        "n_placement_pass": sum(
            1 for o in placement_obligations if o.get("status") == "pass"
        ),
        "n_diff_placement": len(diff_obls),
        "n_mirror_placement": len(mirror_obls),
        "n_shield_obligations": len(shield_results),
        "n_shield_pass": n_shield_pass,
    }

    return {
        "cell": cell_name,
        "status": status,
        "token": token,
        "passed": status == "pass",
        "blocking": blocking,
        "rows": rows,
        "parasitic_obligations": parasitic_obligations,
        "placement_obligations": placement_obligations,
        "shield_obligations": shield_results,
        "score": score,
        "intent_digest": intent.digest,
        "intent_source": intent.source,
        "intent_profile_id": intent.profile_id,
    }


def _fmt_parasitic_pair(row: dict) -> str:
    pair = row.get("evidence", {}).get("pair") or []
    if len(pair) == 2:
        return f"{pair[0]}~{pair[1]}"
    return str(row.get("source", {}).get("name", ""))


def _fmt_placement_devices(row: dict) -> str:
    devs = row.get("evidence", {}).get("devices") or []
    if len(devs) == 2:
        return f"{devs[0]}/{devs[1]}"
    return ""


def format_full_signoff(result: Dict[str, object]) -> str:
    """Render a combined human-readable table for a unified signoff result.

    Two sections -- a PARASITIC section (kind | pair | measured | budget | verdict)
    and a PLACEMENT section (pair | devices | dx | dy | budget | verdict) -- then
    one overall verdict line and the proof-token state (signed digest / WITHHELD).
    """
    cell = str(result.get("cell", "?"))
    rows: List[dict] = list(result.get("rows", []))  # type: ignore[arg-type]
    parasitic_rows = [
        r for r in rows
        if r.get("source_kind") not in (PLACEMENT_SOURCE_KIND, SHIELD_SOURCE_KIND)
    ]
    placement_rows = [r for r in rows if r.get("source_kind") == PLACEMENT_SOURCE_KIND]
    shield_rows = [r for r in rows if r.get("source_kind") == SHIELD_SOURCE_KIND]

    out: List[str] = []
    out.append(f"Unified layout signoff: cell={cell}")
    digest = result.get("intent_digest")
    if digest:
        out.append(
            f"  intent: digest={digest} "
            f"source={result.get('intent_source', 'auto')} "
            f"profile={result.get('intent_profile_id') or 'none'}"
        )
    out.append("")

    # ── PARASITIC section ────────────────────────────────────────────────────
    out.append("PARASITIC obligations")
    p_headers = ("kind", "pair", "measured", "budget", "verdict")
    p_table: List[tuple] = []
    for row in parasitic_rows:
        ev = row.get("evidence", {})
        raw_measured = ev.get("measured_imbalance")
        budget = float(ev.get("rel_tol", 0.0))
        status = row.get("status")
        if status == "pass":
            verdict = "PASS"
        elif raw_measured is None:
            verdict = "UNMEAS"
        else:
            verdict = "FAIL"
        measured_str = " n/a   " if raw_measured is None else f"{float(raw_measured):7.4f}"
        p_table.append(
            (
                str(row.get("obligation_kind", "?")),
                _fmt_parasitic_pair(row),
                measured_str,
                f"{budget:7.4f}",
                verdict,
            )
        )
    out.extend(_render_table(p_headers, p_table, "(no declared parasitic budgets)"))
    out.append("")

    # ── PLACEMENT section (diff + mirror device pairs) ───────────────────────
    out.append("PLACEMENT obligations")
    pl_headers = ("kind", "pair", "devices", "dx", "dy", "budget", "verdict")
    pl_table: List[tuple] = []
    for row in placement_rows:
        ev = row.get("evidence", {})
        dx = float(ev.get("dx_imbalance", 0.0))
        dy = float(ev.get("dy_offset", 0.0))
        budget = float(ev.get("rel_tol", 0.0))
        verdict = "PASS" if row.get("status") == "pass" else "FAIL"
        pl_table.append(
            (
                str(ev.get("match_kind", "differential")),
                _fmt_parasitic_pair(row),
                _fmt_placement_devices(row),
                f"{dx:7.4f}",
                f"{dy:7.4f}",
                f"{budget:7.4f}",
                verdict,
            )
        )
    out.extend(_render_table(pl_headers, pl_table, "(no matched device pairs)"))
    out.append("")

    # ── SHIELD section ───────────────────────────────────────────────────────
    if shield_rows:
        out.append("SHIELD obligations")
        sh_headers = ("aggressor", "victim", "shield_net", "verdict")
        sh_table: List[tuple] = []
        for row in shield_rows:
            ev = row.get("evidence", {})
            verdict = "PASS" if row.get("status") == "pass" else "FAIL"
            sh_table.append(
                (
                    str(ev.get("aggressor", "?")),
                    str(ev.get("victim", "?")),
                    str(ev.get("shield_net", "gnd")),
                    verdict,
                )
            )
        out.extend(_render_table(sh_headers, sh_table, "(no declared shields)"))
        out.append("")

    # ── Overall verdict + token ──────────────────────────────────────────────
    status = str(result.get("status", "?"))
    passed = bool(result.get("passed", False))
    blocking = list(result.get("blocking", []))  # type: ignore[arg-type]
    score = result.get("score", {}) or {}
    verdict_word = "CERTIFIED" if passed else "NOT CERTIFIED"
    out.append(f"Overall verdict: {verdict_word} (status={status})")
    if isinstance(score, dict) and "combined" in score:
        out.append(
            "Score: combined={:.4f} (parasitic={:.4f}, placement_pass={:.4f})".format(
                float(score["combined"]),
                float(score.get("parasitic_score", 0.0)),
                float(score.get("placement_pass_fraction", 0.0)),
            )
        )
    if blocking:
        out.append(f"Blocking obligations: {', '.join(blocking)}")

    token = result.get("token")
    if token is None:
        out.append(
            "Proof token: WITHHELD (a required parasitic OR placement obligation "
            "failed; withheld exactly like a DRC violation -- never fabricated)"
        )
    else:
        digest = ""
        if isinstance(token, dict):
            digest = str(token.get("trace_digest", ""))
        out.append(f"Proof token: SIGNED (trace_digest={digest})")

    return "\n".join(out)


def _render_table(headers, table_rows, empty_msg: str) -> List[str]:
    """Render an aligned text table; returns the header + separator + rows lines."""
    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, cellv in enumerate(tr):
            widths[i] = max(widths[i], len(str(cellv)))

    def _line(cols) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))

    sep = "  ".join("-" * w for w in widths)
    lines = [_line(headers), sep]
    if table_rows:
        for tr in table_rows:
            lines.append(_line(tr))
    else:
        lines.append(empty_msg)
    return lines


__all__ = [
    "full_layout_signoff",
    "format_full_signoff",
    "placement_symmetry_rows",
    "PLACEMENT_SOURCE_KIND",
]
