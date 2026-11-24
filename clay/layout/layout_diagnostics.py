"""Quantitative pre-PEX layout diagnostics.

Per user directive 2026-05-28: the global constraints checker only said
"violates rule X with bbox Y"; agents need ACTIONABLE diagnostics:

  * predicted Magic-merge pairs (which two nets WILL extract as one)
  * worst spacing per layer + which two polygons
  * connectivity per net (are device terminals reachable through the routes)
  * rail tag completeness

These are computed from the cell's geometry BEFORE running Magic; an agent
can fix predicted merges in seconds instead of discovering them after a
60-second extraction.

API:
    report = compute_diagnostics(routed_cell)
    # report["predicted_merges"]   list of {layer,nets,overlap_area_nm2,bbox_a,bbox_b}
    # report["min_spacing"]        per-layer worst spacing + offending polygons
    # report["disconnected_nets"]  per-net connected-component count > 1
    # report["rail_tag"]           {"vdd","gnd": {"tagged": bool, "n_polygons": int}}
    # report["summary"]            quick numbers for cross-cell tracking

Also exposes load_layout_knowledge() to read the YAML lessons file (so
layout drivers + agents can iterate over the captured fixes by tag).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from clay.layout.global_constraints import (
    BBox, _iter_tagged, _rail_net_aliases, _bbox_overlap,
)

_KNOWLEDGE_YAML = Path(__file__).parent / "layout_knowledge.yaml"
_PDK_INTENT_DIR = Path(__file__).parent.parent / "pdk"


# ─── knowledge access ──────────────────────────────────────────────────────
# Process-AGNOSTIC principles live in clay/layout/layout_knowledge.yaml.
# Process-BOUND gotchas live per-PDK in clay/pdk/<pdk_id>_layout_intent.yaml.
# Agents typically want all_layout_knowledge(pdk_id) -- both merged with
# source attribution so they know which apply universally vs only to a PDK.

def load_layout_knowledge(path: Optional[Path] = None) -> List[dict]:
    """Process-agnostic layout principles (clay/layout/layout_knowledge.yaml)."""
    p = path or _KNOWLEDGE_YAML
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    return list(doc.get("lessons") or [])


def load_pdk_layout_intent(pdk_id: str) -> List[dict]:
    """PDK-specific layout intent / gotchas
    (clay/pdk/<pdk_id>_layout_intent.yaml). Returns [] if no file for that
    PDK yet -- treat absent as "no PDK-specific intent captured yet"."""
    p = _PDK_INTENT_DIR / f"{pdk_id}_layout_intent.yaml"
    if not p.is_file():
        return []
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    return list(doc.get("intent") or doc.get("lessons") or [])


def all_layout_knowledge(pdk_id: Optional[str] = None) -> List[dict]:
    """Both the process-agnostic principles AND the PDK-specific intent,
    tagged with origin so callers can tell them apart."""
    out = []
    for entry in load_layout_knowledge():
        out.append({**entry, "origin": "process-agnostic"})
    if pdk_id:
        for entry in load_pdk_layout_intent(pdk_id):
            out.append({**entry, "origin": f"pdk:{pdk_id}"})
    return out


def lessons_by_tag(tag: str, pdk_id: Optional[str] = None) -> List[dict]:
    """Filter lessons by tag (e.g. 'magic', 'routing'), across both stores."""
    return [l for l in all_layout_knowledge(pdk_id) if tag in (l.get("tags") or [])]


# ─── quantitative diagnostics ──────────────────────────────────────────────

def _overlap_area(a: BBox, b: BBox) -> float:
    dx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    dy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return dx * dy


def _gap(a: BBox, b: BBox) -> float:
    """Manhattan gap between two non-overlapping bboxes (0 if touching/overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def predicted_magic_merges(
    routed_cell: Any,
    *,
    touch_tol_nm: float = 0.5,
    rail_params: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Same-layer different-net polygon pairs that Magic ext2spice will
    extract as ONE net (overlap > 0 or gap <= touch_tol_nm). The single
    biggest pre-PEX fail predictor; if this list is non-empty BEFORE running
    Magic, the post-sim will be non-physical."""
    vdd_nets, gnd_nets = _rail_net_aliases(rail_params or {})
    rail_nets = vdd_nets | gnd_nets

    by_layer: Dict[str, List[Tuple[str, BBox]]] = {}
    for net, layer, b in _iter_tagged(routed_cell):
        by_layer.setdefault(layer, []).append((net, b))

    merges: List[Dict[str, Any]] = []
    for layer, items in by_layer.items():
        n = len(items)
        for i in range(n):
            net_a, ba = items[i]
            for j in range(i + 1, n):
                net_b, bb = items[j]
                if net_a == net_b:
                    continue
                # rail polys are exempt from "merge" with themselves; signal vs
                # rail is already covered by no_signal_crosses_rails but we
                # report it here too with the overlap area for actionable fix.
                area = _overlap_area(ba, bb)
                gap = _gap(ba, bb) if area == 0 else 0.0
                if area > 0 or gap <= touch_tol_nm:
                    merges.append({
                        "layer": layer,
                        "nets": sorted([net_a, net_b]),
                        "overlap_area_nm2": area,
                        "gap_nm": gap,
                        "bbox_a": ba,
                        "bbox_b": bb,
                        "involves_rail": bool({net_a, net_b} & rail_nets),
                    })
    merges.sort(key=lambda m: -m["overlap_area_nm2"])
    return merges


def min_spacing_per_layer(
    routed_cell: Any,
    *,
    rail_params: Optional[dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-layer worst-case spacing between different-net non-overlapping polys."""
    by_layer: Dict[str, List[Tuple[str, BBox]]] = {}
    for net, layer, b in _iter_tagged(routed_cell):
        by_layer.setdefault(layer, []).append((net, b))

    out: Dict[str, Dict[str, Any]] = {}
    for layer, items in by_layer.items():
        worst: Optional[Tuple[float, str, str, BBox, BBox]] = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                na, ba = items[i]
                nb, bb = items[j]
                if na == nb or _bbox_overlap(ba, bb):
                    continue
                g = _gap(ba, bb)
                if worst is None or g < worst[0]:
                    worst = (g, na, nb, ba, bb)
        if worst is not None:
            out[layer] = {
                "min_spacing_nm": worst[0],
                "nets": [worst[1], worst[2]],
                "bbox_a": worst[3],
                "bbox_b": worst[4],
            }
    return out


def connectivity_per_net(routed_cell: Any) -> Dict[str, int]:
    """Per-net count of disjoint geometric components (1 = fully connected; >1 = split).

    Builds a graph over each net's polygons, edges = bbox overlap, layer
    transitions via shared bbox (a rough proxy for "is there a via stack" --
    not as strict as a real connectivity extractor, but catches gross splits).
    """
    by_net: Dict[str, List[Tuple[str, BBox]]] = {}
    for net, layer, b in _iter_tagged(routed_cell):
        by_net.setdefault(net, []).append((layer, b))

    out: Dict[str, int] = {}
    for net, items in by_net.items():
        n = len(items)
        parent = list(range(n))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i in range(n):
            for j in range(i + 1, n):
                # any overlap (same layer OR cross-layer same x/y region) ->
                # treat as connected. This over-connects vias but matches the
                # spirit of "is the net one piece".
                if _bbox_overlap(items[i][1], items[j][1]):
                    union(i, j)
        out[net] = len({find(i) for i in range(n)})
    return out


def rail_tag_completeness(
    routed_cell: Any,
    *,
    rail_params: Optional[dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """Are vdd/gnd rails tagged at all? How many polygons per rail net?"""
    vdd_nets, gnd_nets = _rail_net_aliases(rail_params or {})
    vdd_count = sum(1 for n, _, _ in _iter_tagged(routed_cell) if n in vdd_nets)
    gnd_count = sum(1 for n, _, _ in _iter_tagged(routed_cell) if n in gnd_nets)
    return {
        "vdd": {"tagged": vdd_count > 0, "n_polygons": vdd_count,
                "accepted_names": sorted(vdd_nets)},
        "gnd": {"tagged": gnd_count > 0, "n_polygons": gnd_count,
                "accepted_names": sorted(gnd_nets)},
    }


def compute_diagnostics(
    routed_cell: Any,
    *,
    touch_tol_nm: float = 0.5,
    rail_params: Optional[dict] = None,
) -> Dict[str, Any]:
    """Full quantitative pre-PEX report. The single API agents should call
    after building a layout and BEFORE running Magic.
    """
    merges = predicted_magic_merges(
        routed_cell, touch_tol_nm=touch_tol_nm, rail_params=rail_params)
    spacing = min_spacing_per_layer(routed_cell, rail_params=rail_params)
    conn = connectivity_per_net(routed_cell)
    rail = rail_tag_completeness(routed_cell, rail_params=rail_params)
    disconnected = {net: k for net, k in conn.items() if k > 1}
    return {
        "predicted_merges": merges,
        "min_spacing_per_layer": spacing,
        "disconnected_nets": disconnected,
        "rail_tag": rail,
        "summary": {
            "n_predicted_merges": len(merges),
            "n_merges_involving_rail": sum(1 for m in merges if m["involves_rail"]),
            "worst_overlap_nm2": merges[0]["overlap_area_nm2"] if merges else 0.0,
            "n_disconnected_nets": len(disconnected),
            "rails_tagged": rail["vdd"]["tagged"] and rail["gnd"]["tagged"],
            "n_layers": len(spacing),
            "n_nets": len(conn),
        },
    }


def format_diagnostics(report: Dict[str, Any], *, max_items: int = 5) -> str:
    """Compact human/agent-readable rendering."""
    s = report["summary"]
    lines = [
        "Layout pre-PEX diagnostics",
        f"  predicted Magic merges : {s['n_predicted_merges']:>3d}  "
        f"(worst overlap {s['worst_overlap_nm2']:.0f} nm^2, "
        f"{s['n_merges_involving_rail']} involve a rail)",
        f"  disconnected nets      : {s['n_disconnected_nets']:>3d}",
        f"  rails tagged           : {'yes' if s['rails_tagged'] else 'NO -- tag rails as vdd/gnd or post-PEX will float'}",
        f"  nets x layers          : {s['n_nets']} nets across {s['n_layers']} layers",
    ]
    if report["predicted_merges"]:
        lines.append(f"  top predicted merges (up to {max_items}):")
        for m in report["predicted_merges"][:max_items]:
            lines.append(
                f"    - layer {m['layer']}: nets {m['nets'][0]} & {m['nets'][1]} "
                f"overlap_area={m['overlap_area_nm2']:.0f} nm^2 gap={m['gap_nm']:.1f}"
            )
    if report["disconnected_nets"]:
        lines.append(f"  disconnected nets (up to {max_items}):")
        for net, k in list(report["disconnected_nets"].items())[:max_items]:
            lines.append(f"    - {net}: {k} disjoint fragments")
    return "\n".join(lines)


__all__ = [
    "compute_diagnostics",
    "format_diagnostics",
    "predicted_magic_merges",
    "min_spacing_per_layer",
    "connectivity_per_net",
    "rail_tag_completeness",
    "load_layout_knowledge",
    "lessons_by_tag",
]
