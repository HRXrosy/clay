"""Python bridge for the Rust ``net-connectivity`` union-find kernel (R170, BL-170E).

Invokes the ``clay_layout_gds_bin net-connectivity`` subcommand (JSON in / JSON
out) to run the geometric net-connectivity / union-find that every multi-block
routing co-clean relies on: given a set of placed shapes ``(net, (layer,
datatype), bbox)``, a same-layer touch rule and a set of via layers that bridge
two metals, compute which shapes are electrically merged (the connected
components) and detect inter-net SHORTS -- any component containing more than
one distinct net.

This is the numeric core of the ``verify_distinct_nets`` geometric union-find in
``clay/layout/shape_router_v168.py`` and ``shape_detailed_router_v169.py``: two
polygons connect when their bounding boxes touch / overlap (inclusive AABB
intersection) AND either they share the same ``(layer, datatype)`` (same-layer
metal merge) OR at least one of them is on a via layer (a via bridges two
metals). The union-find uses path halving (``parent[i] = parent[parent[i]]``)
and union-toward-the-second-root (``parent[ri] = rj``), exactly as the Python
``verify_distinct_nets`` does, and the pairwise scan runs ``i`` over the shapes
and ``j`` over the shapes after ``i`` so the resulting root labels match
key-for-key.

Net identities are arbitrary hashable labels (typically net-name strings). The
bridge interns each label to a stable ``usize`` index in first-seen order (the
Rust kernel is index-based, name-agnostic) and maps results back to the original
labels, so a caller works purely in net names.

The pure-Python reference below mirrors the Rust kernel bit-for-bit; the Rust
kernel is the fast path. This bridge is thin: it delegates binary discovery +
subprocess invocation + ``rust_available`` to the shared ``clay/_rust_bridge_core``
helpers. The pure-Python fallback is used automatically when the Rust binary is
unavailable, and produces identical results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

# Reuse the binary discovery from the existing GDS subprocess bridge so all
# clay_layout bridges agree on which build of the CLI to use.
from .binary_gds_subprocess import _find_binary, BinaryGdsSubprocessError
from .._rust_bridge_core import invoke_layout_kernel, layout_rust_available

# A shape is (net_label, (layer, datatype), (x0, y0, x1, y1)).
Bbox = Tuple[int, int, int, int]
LayerKey = Tuple[int, int]
Shape = Tuple[Hashable, LayerKey, Bbox]


# --- pure-Python reference (exact mirror of the Rust kernel) ------------------


def _normalise_bbox(b: Bbox) -> Bbox:
    """Order corners so ``x_min <= x_max`` and ``y_min <= y_max``. Mirrors the
    Rust ``Bbox::new``."""
    x0, y0, x1, y1 = b
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def py_touch_or_overlap(a: Bbox, b: Bbox) -> bool:
    """Inclusive AABB touch/overlap test (edges/corners touching counts as
    connected). Mirrors the Rust ``touch_or_overlap`` and the Python
    ``_touch_or_overlap`` in the routing modules:
    ``a.x0 <= b.x1 and b.x0 <= a.x1 and a.y0 <= b.y1 and b.y0 <= a.y1``."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def py_layer_connects(a: LayerKey, b: LayerKey, via_layers: Sequence[LayerKey]) -> bool:
    """Whether two touching shapes are electrically connected: same
    ``(layer, datatype)`` (same-layer metal), or at least one is on a via layer
    (a via bridges two metals). Mirrors the Rust ``layer_connects`` and the
    ``same_layer or via_bridge`` clause in ``verify_distinct_nets``."""
    if a == b:
        return True
    return a in via_layers or b in via_layers


def py_connected_components(
    shapes: Sequence[Shape],
    via_layers: Sequence[LayerKey] = (),
) -> dict:
    """Compute connected components over ``shapes`` and detect inter-net shorts.

    ``shapes`` is a sequence of ``(net_label, (layer, datatype), bbox)`` triples;
    ``via_layers`` is the set of via ``(layer, datatype)`` keys. Returns a dict
    with keys:

      * ``component_of`` -- ``{net_label: component_root}`` (the min root over
        that net's shapes);
      * ``merged_pairs`` -- sorted list of ``[a, b]`` net-label pairs that share
        a component (the inter-net shorts), deduplicated;
      * ``distinct`` -- ``True`` iff ``merged_pairs`` is empty;
      * ``shape_root`` -- per-shape final component root, in input order;
      * ``short_count`` -- ``len(merged_pairs)``.

    Mirrors the Rust ``connected_components`` + the CLI ``run_net_connectivity``
    body, and the Python ``verify_distinct_nets`` union-find. Net labels are
    interned to first-seen ``usize`` indices for the union-find (so the Rust path
    is index-based); the returned dict uses the original labels."""
    via_set = set(via_layers)

    # Intern net labels to indices in first-seen order (matches the Rust bridge
    # payload ordering). The union-find roots are SHAPE indices, not net ids.
    items: List[Tuple[int, LayerKey, Bbox]] = []
    label_of: Dict[Hashable, int] = {}
    labels: List[Hashable] = []
    for net_label, layer, bbox in shapes:
        if net_label not in label_of:
            label_of[net_label] = len(labels)
            labels.append(net_label)
        items.append((label_of[net_label], tuple(layer), _normalise_bbox(bbox)))

    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        _ni, ki, bi = items[i]
        for j in range(i + 1, n):
            _nj, kj, bj = items[j]
            if not py_touch_or_overlap(bi, bj):
                continue
            if py_layer_connects(ki, kj, via_set):
                union(i, j)

    shape_root = [find(idx) for idx in range(n)]

    net_components: Dict[int, set] = {}
    root_to_nets: Dict[int, set] = {}
    for idx, (net_id, _k, _b) in enumerate(items):
        r = shape_root[idx]
        net_components.setdefault(net_id, set()).add(r)
        root_to_nets.setdefault(r, set()).add(net_id)

    # merged_pairs over net IDS, then mapped back to labels and sorted.
    pair_ids: set = set()
    for _r, nets in root_to_nets.items():
        if len(nets) > 1:
            ordered = sorted(nets)
            for a in range(len(ordered)):
                for b in range(a + 1, len(ordered)):
                    pair_ids.add((ordered[a], ordered[b]))

    merged_pairs = sorted(
        [labels[a], labels[b]] for (a, b) in pair_ids
    )

    component_of = {labels[net]: min(comps) for net, comps in net_components.items()}
    distinct = len(merged_pairs) == 0
    return {
        "component_of": component_of,
        "merged_pairs": merged_pairs,
        "distinct": distinct,
        "shape_root": shape_root,
        "short_count": len(merged_pairs),
    }


# --- Rust CLI invocation -----------------------------------------------------


def _rust_compute(
    shapes: Sequence[Shape],
    via_layers: Sequence[LayerKey],
    *,
    binary_path: Optional[Path] = None,
    timeout_sec: float = 60.0,
) -> dict:
    """Invoke the Rust ``net-connectivity`` subcommand; return a result dict in
    the same shape as ``py_connected_components`` (labels restored).

    Raises ``BinaryGdsSubprocessError`` if the binary is missing or the
    subprocess fails."""
    # Intern net labels to first-seen indices (the Rust kernel is index-based).
    label_of: Dict[Hashable, int] = {}
    labels: List[Hashable] = []
    json_shapes = []
    for net_label, layer, bbox in shapes:
        if net_label not in label_of:
            label_of[net_label] = len(labels)
            labels.append(net_label)
        x0, y0, x1, y1 = bbox
        json_shapes.append(
            {
                "net_id": int(label_of[net_label]),
                "layer": [int(layer[0]), int(layer[1])],
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
            }
        )

    payload = {
        "shapes": json_shapes,
        "via_layers": [[int(v[0]), int(v[1])] for v in via_layers],
    }
    raw = invoke_layout_kernel(
        "net-connectivity",
        payload,
        find_binary=_find_binary,
        error_cls=BinaryGdsSubprocessError,
        binary_path=binary_path,
        timeout_sec=timeout_sec,
        missing_binary_hint="  (cd crates/clay_layout && cargo build --release)",
    )

    # Map index-based results back to the original net labels.
    component_of = {labels[net_id]: root for net_id, root in raw["component_of"]}
    merged_pairs = sorted(
        [labels[a], labels[b]] for (a, b) in raw["merged_pairs"]
    )
    return {
        "component_of": component_of,
        "merged_pairs": merged_pairs,
        "distinct": bool(raw["distinct"]),
        "shape_root": list(raw["shape_root"]),
        "short_count": int(raw["short_count"]),
    }


# --- public API (Rust with pure-Python fallback) -----------------------------


def verify_connectivity(
    shapes: Sequence[Shape],
    *,
    via_layers: Sequence[LayerKey] = (),
    prefer_rust: bool = True,
    binary_path: Optional[Path] = None,
) -> dict:
    """Compute connected components + detect inter-net shorts over placed shapes.

    Args:
        shapes: sequence of ``(net_label, (layer, datatype), bbox)`` triples,
            where ``bbox`` is ``(x0, y0, x1, y1)`` in DBU (corners need not be
            ordered). ``net_label`` is any hashable (typically a net-name string).
        via_layers: the set of via ``(layer, datatype)`` keys; a shape on any of
            these bridges two metal layers.

    Returns a dict with keys ``component_of`` (``{net_label: root}``),
    ``merged_pairs`` (sorted ``[a, b]`` shorted-net pairs), ``distinct`` (bool),
    ``shape_root`` (per-shape root, input order) and ``short_count`` (int).

    Uses the Rust CLI when available and ``prefer_rust`` is True; otherwise falls
    back to the pure-Python reference. The two paths produce identical results.
    """
    if prefer_rust:
        if binary_path is None:
            try:
                from clay._native import load_native

                eng = load_native("_engine_cpp")
                if eng is not None and hasattr(eng, "verify_net_connectivity_cpp"):
                    return eng.verify_net_connectivity_cpp(shapes, via_layers)
            except Exception:
                pass
        try:
            return _rust_compute(shapes, via_layers, binary_path=binary_path)
        except BinaryGdsSubprocessError:
            pass  # fall through to pure Python
    return py_connected_components(shapes, via_layers)


def rust_available(binary_path: Optional[Path] = None) -> bool:
    """True if the Rust CLI binary is present on disk (existence only; not version/freshness/executability verified)."""
    return layout_rust_available(_find_binary, binary_path)


__all__ = [
    "Bbox",
    "LayerKey",
    "Shape",
    "py_touch_or_overlap",
    "py_layer_connects",
    "py_connected_components",
    "verify_connectivity",
    "rust_available",
]
