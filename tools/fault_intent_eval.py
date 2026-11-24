"""Formal ablation study for the error-intent solver (clay.layout.fault_intent).

EXPERIMENT harness (see docs/superpowers/round-experiment-fault-intent.md). Not a
production path; it quantifies whether the solver delivers a real, attributable
effect before any integration decision.

The general problem (abstracted)
================================
LAYOUT FAULT LOCALISATION & EXPLANATION. Given a violation set V (DRC/LVS) on a
layout produced by discrete engineering decisions D under declared intent I,
produce a minimal explanation E = (root decisions D* subset D, violated intents
I* subset I, corrective levers L) such that acting on D* removes V. This is an
inverse-attribution problem; the baseline feedback (raw V with coordinates) lives
in the wrong space for a decision-level agent.

Metrics (all MEASURED, none asserted)
-------------------------------------
  compression rho      = |V| / |root causes|   (how much the agent's reasoning load shrinks)
  localisation_acc     = fraction of cases where #root causes == #ground-truth columns
                         AND each root recovers its column's net set  (synthetic only;
                         ground truth is known)
  intent_acc           = fraction of root causes labelled with the correct violated intent
  cf_soundness         = fraction of root causes whose counterfactual is geometrically
                         exact (validate_counterfactuals: actually remove the named net,
                         re-extract, confirm the claimed coverage)
  latency_ms           = solver wall time

Ablation arms (what each component of the diagnosis buys)
---------------------------------------------------------
  A0 raw          : the flat violation list (today's feedback). reasoning items = |V|.
  A1 +cluster     : group symptoms into root decisions. reasoning items = |roots|; adds rho + localisation.
  A2 +intent      : label each root with the violated intent. adds intent_acc.
  A3 +counterfactual: add the verified single-step counterfactual + levers. adds cf_soundness.

Corpora
-------
  SYNTHETIC (ground truth known): controlled cells with K columns x M co-located nets,
    intent in {rail, diff, generic}. Sweeps K, M, intent. Real union-find shorts.
  REAL: the production circuits' generic routes (intrinsic symptoms), ecological validity.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from clay.layout.m2_crossing_route import M1_KEY  # noqa: E402
from clay.layout.routed_cell_types import check_shorts  # noqa: E402
from clay.layout.fault_intent import (  # noqa: E402
    diagnose,
    validate_counterfactuals,
)

Bbox = Tuple[int, int, int, int]
TaggedShape = Tuple[str, Tuple[int, int], Bbox]

_COLUMN_SPACING_NM = 20000  # >> fault_intent._COLUMN_TOL_NM so columns are distinct


class _Routed:
    """Minimal RoutedCell-shaped object: just .tagged + .shorts (what diagnose reads)."""

    def __init__(self, tagged: List[TaggedShape]):
        self.tagged = tagged
        self.shorts = check_shorts(tagged)
        self.polys = []
        self.labels = {}


class _Criteria:
    def __init__(self, device_pairs):
        self.device_pairs = tuple(device_pairs)
        self.matched_pairs = tuple(device_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# synthetic generator (ground truth known)
# ─────────────────────────────────────────────────────────────────────────────
def synth_case(
    n_columns: int, nets_per_column: int, intent_kind: str
) -> Tuple[_Routed, List[dict], _Criteria]:
    """Build a routed-like cell with ``n_columns`` columns, each a stack of
    ``nets_per_column`` co-located (overlapping) M1 bars that union-find merges
    into shorts. Returns (routed, ground_truth, criteria).

    intent_kind: "rail" -> one bar is vss (signal-to-rail short); "diff" -> two
    bars are a declared diff pair; "generic" -> plain signal nets."""
    tagged: List[TaggedShape] = []
    truth: List[dict] = []
    diff_pairs: List[Tuple[str, str]] = []
    for c in range(n_columns):
        x = c * _COLUMN_SPACING_NM
        nets: List[str] = []
        if intent_kind == "rail":
            nets = ["vss"] + [f"s{c}_{i}" for i in range(nets_per_column - 1)]
            kind = "rail isolation"
        elif intent_kind == "diff":
            nets = [f"inp{c}", f"inn{c}"] + [f"s{c}_{i}" for i in range(nets_per_column - 2)]
            diff_pairs.append((f"inp{c}", f"inn{c}"))
            kind = "diff-pair isolation"
        else:
            nets = [f"n{c}_{i}" for i in range(nets_per_column)]
            kind = "net_disjoint"
        # overlapping vertical M1 bars at the same x -> all touch -> all merge.
        for net in nets:
            tagged.append((net, M1_KEY, (x - 100, 0, x + 100, 2000)))
        truth.append({"x": x, "nets": set(nets), "intent": kind})
    return _Routed(tagged), truth, _Criteria(diff_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# per-case metrics
# ─────────────────────────────────────────────────────────────────────────────
def _intent_correct(label: str, kind: str) -> bool:
    label = label.lower()
    if kind == "rail isolation":
        return "rail" in label
    if kind == "diff-pair isolation":
        return "diff-pair" in label
    return "net_disjoint" in label


def eval_case(routed: _Routed, truth: Optional[List[dict]], criteria) -> dict:
    t0 = time.perf_counter()
    dg = diagnose(routed, criteria=criteria, cell="synth")
    latency_ms = (time.perf_counter() - t0) * 1e3
    n_v = dg.n_symptoms
    n_roots = len(dg.root_causes)
    rho = (n_v / n_roots) if n_roots else float("nan")

    # localisation (synthetic only): #roots == #truth columns AND nets recovered.
    loc_ok = None
    intent_ok_frac = None
    if truth is not None:
        loc_ok = (n_roots == len(truth))
        if loc_ok:
            truth_x = sorted(t["x"] for t in truth)
            root_x = sorted(_root_x(rc) for rc in dg.root_causes)
            # match roots to truth columns by nearest x; check net recovery + intent.
            matched = 0
            intent_hits = 0
            for rc in dg.root_causes:
                rx = _root_x(rc)
                tr = min(truth, key=lambda t: abs(t["x"] - rx))
                if set(rc.nets_involved) >= tr["nets"]:
                    matched += 1
                if _intent_correct(rc.intent_violated, tr["intent"]):
                    intent_hits += 1
            loc_ok = (matched == len(truth))
            intent_ok_frac = intent_hits / n_roots if n_roots else 0.0
        else:
            intent_ok_frac = 0.0

    cf_soundness, _ = validate_counterfactuals(routed, dg)
    return {
        "n_v": n_v,
        "n_roots": n_roots,
        "rho": rho,
        "loc_ok": loc_ok,
        "intent_acc": intent_ok_frac,
        "cf_soundness": cf_soundness,
        "latency_ms": latency_ms,
    }


def _root_x(rc) -> int:
    # decision string is "column@x≈<int> ..."; parse the int.
    import re
    m = re.search(r"x≈(-?\d+)", rc.decision)
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────────────────────────
# runs
# ─────────────────────────────────────────────────────────────────────────────
def run_synthetic() -> List[dict]:
    rows = []
    for kind in ("rail", "diff", "generic"):
        for k in (1, 2, 4, 8):
            for m in (2, 3, 4):
                routed, truth, crit = synth_case(k, m, kind)
                r = eval_case(routed, truth, crit)
                r.update({"kind": kind, "columns": k, "nets_per_col": m})
                rows.append(r)
    return rows


def run_real() -> List[dict]:
    import os
    from clay.flow.spec import discover_registry
    from clay.flow.step import LayoutStep
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.generic_placement import place_generic, parse_cir_mos_records
    from clay.layout.route.route_cell import route_cell
    from clay.layout.guide_intent import build_criteria, known_circuits

    reg = discover_registry()
    known = set(known_circuits())
    rows = []
    for circ in ("fc_ota", "two_stage", "ldo", "bandgap"):
        spec = reg.get(circ)
        if spec is None:
            continue
        cd = LayoutStep._parse_cell_def(spec)  # the FLOW's own cell resolution
        if cd is None:
            continue
        intent = layout_intent_from_cell_def(cd)
        # the registry-resolved source .cir (general; never a hardcoded guess).
        src = spec.source_cir if (spec.source_cir and os.path.exists(spec.source_cir)) else None
        p = place_generic(cd, intent, None, source_cir=src)
        recs = parse_cir_mos_records(src) if src else {}
        # fault-intent eval works on possibly-incomplete routes -> do not raise.
        # Route through the GENERAL production router (route_cell); the legacy
        # route_generic planner is retired.
        routed = route_cell(
            p, cd, intent, spec.pdk,
            require_complete=False, device_records=recs,
        ).to_routed_cell()
        crit = build_criteria(circ) if circ in known else None
        r = eval_case(routed, None, crit)
        r.update({"circuit": circ, "tagged": len(routed.tagged)})
        rows.append(r)
    return rows


def _agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and r[key] != r[key])]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    syn = run_synthetic()
    print("=" * 78)
    print("SYNTHETIC corpus (ground truth known) — per intent kind")
    print("=" * 78)
    print(f"{'kind':9} {'cases':5} {'mean|V|':8} {'mean roots':10} {'rho':6} "
          f"{'loc_acc':8} {'intent_acc':10} {'cf_sound':9}")
    for kind in ("rail", "diff", "generic"):
        rs = [r for r in syn if r["kind"] == kind]
        loc = sum(1 for r in rs if r["loc_ok"]) / len(rs)
        print(f"{kind:9} {len(rs):<5d} {_agg(rs,'n_v'):<8.1f} {_agg(rs,'n_roots'):<10.2f} "
              f"{_agg(rs,'rho'):<6.2f} {loc*100:<7.0f}% {_agg(rs,'intent_acc')*100:<9.0f}% "
              f"{_agg(rs,'cf_soundness')*100:<8.0f}%")
    loc_all = sum(1 for r in syn if r["loc_ok"]) / len(syn)
    print(f"{'ALL':9} {len(syn):<5d} {_agg(syn,'n_v'):<8.1f} {_agg(syn,'n_roots'):<10.2f} "
          f"{_agg(syn,'rho'):<6.2f} {loc_all*100:<7.0f}% {_agg(syn,'intent_acc')*100:<9.0f}% "
          f"{_agg(syn,'cf_soundness')*100:<8.0f}%")

    print("\n" + "=" * 78)
    print("ABLATION — what each diagnosis component buys (synthetic, mean over all cases)")
    print("=" * 78)
    items_a0 = _agg(syn, "n_v")
    items_a3 = _agg(syn, "n_roots")
    print(f"{'arm':22} {'reasoning items':16} {'attribution':12} {'intent':8} {'verified-cf':11}")
    print(f"{'A0 raw list':22} {items_a0:<16.1f} {'no':12} {'no':8} {'no':11}")
    print(f"{'A1 +cluster':22} {items_a3:<16.2f} {'column':12} {'no':8} {'no':11}")
    print(f"{'A2 +intent':22} {items_a3:<16.2f} {'column':12} {_agg(syn,'intent_acc')*100:<7.0f}% {'no':11}")
    print(f"{'A3 +counterfactual':22} {items_a3:<16.2f} {'column':12} {_agg(syn,'intent_acc')*100:<7.0f}% "
          f"{_agg(syn,'cf_soundness')*100:<10.0f}%")
    print(f"\nheadline: reasoning load {items_a0:.1f} raw violations -> {items_a3:.2f} root "
          f"causes (compression {_agg(syn,'rho'):.2f}x), localisation {loc_all*100:.0f}%, "
          f"intent {_agg(syn,'intent_acc')*100:.0f}%, counterfactual soundness {_agg(syn,'cf_soundness')*100:.0f}%")

    print("\n" + "=" * 78)
    print("LATENCY vs problem size (synthetic)")
    print("=" * 78)
    for k in (1, 2, 4, 8):
        rs = [r for r in syn if r["columns"] == k]
        print(f"  {k} columns: mean |V|={_agg(rs,'n_v'):.0f}  latency={_agg(rs,'latency_ms'):.2f} ms")

    print("\n" + "=" * 78)
    print("REAL circuits (generic route, intrinsic symptoms) — ecological validity")
    print("=" * 78)
    real = run_real()
    print(f"{'circuit':10} {'tagged':7} {'|V|':5} {'roots':6} {'rho':6} {'cf_sound':9}")
    for r in real:
        rho = f"{r['rho']:.2f}" if r["rho"] == r["rho"] else "n/a"
        print(f"{r['circuit']:10} {r['tagged']:<7d} {r['n_v']:<5d} {r['n_roots']:<6d} "
              f"{rho:<6} {r['cf_soundness']*100:<8.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
