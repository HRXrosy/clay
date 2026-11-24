"""Python bridge for the Rust `net-wirelength` layout numeric kernel (R155, BL-155C).

Invokes the `clay_layout_gds_bin net-wirelength` subcommand (JSON in / JSON out)
to compute placement-cost proxies:

  * ``net_total_span``       — HPWL (half-perimeter wirelength) of a net's pin bbox
  * ``net_max_pin_span``     — max pairwise Manhattan distance between distinct pins
  * ``netlist_total_hpwl``   — sum of ``net_total_span`` over all nets
  * ``count_overlapping_pairs`` — number of strictly-overlapping placed-bbox pairs

The Rust kernel is a numeric port of the corresponding helpers in
``clay/layout/connectivity_placement.py``. This bridge accepts the same
primitive inputs the Rust side does (a net = list of ``(x, y)`` pin points; a
bbox = ``(x0, y0, x1, y1)``) and provides a pure-Python fallback that is
byte-for-byte equal to the original source functions, used automatically when
the Rust binary is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Reuse the binary discovery from the existing GDS subprocess bridge so both
# bridges agree on which build of the CLI to use.
from .binary_gds_subprocess import _find_binary, BinaryGdsSubprocessError
from .._rust_bridge_core import invoke_layout_kernel, layout_rust_available

PinPoint = Tuple[int, int]
Bbox = Tuple[int, int, int, int]


# --- pure-Python reference (exact mirror of connectivity_placement.py) -------


def py_net_total_span(pins: Sequence[PinPoint]) -> int:
    """HPWL of a net's pin bounding box. Mirrors ``net_total_span``."""
    pts = list({(p[0], p[1]) for p in pins})
    if len(pts) < 2:
        return 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def py_net_max_pin_span(pins: Sequence[PinPoint]) -> int:
    """Max pairwise Manhattan distance between distinct pins.

    Mirrors ``net_max_pin_span`` (the exact O(n^2) pairwise scan)."""
    pts = sorted({(p[0], p[1]) for p in pins})
    best = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            if d > best:
                best = d
    return best


def py_netlist_total_hpwl(nets: Sequence[Sequence[PinPoint]]) -> int:
    """Sum of ``py_net_total_span`` over all nets. Mirrors ``netlist_total_hpwl``."""
    return sum(py_net_total_span(net) for net in nets)


def py_bboxes_overlap(a: Bbox, b: Bbox) -> bool:
    """Strict interior-intersection test. Mirrors ``_bboxes_overlap``
    (touching edges/corners are NOT an overlap)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def py_count_overlapping_pairs(bboxes: Sequence[Bbox]) -> int:
    """Number of strictly-overlapping unordered bbox pairs. Mirrors the
    pairwise scan in ``find_overlaps``."""
    count = 0
    n = len(bboxes)
    for i in range(n):
        for j in range(i + 1, n):
            if py_bboxes_overlap(bboxes[i], bboxes[j]):
                count += 1
    return count


# --- Rust CLI invocation -----------------------------------------------------


def _rust_compute(
    nets: Sequence[Sequence[PinPoint]],
    bboxes: Sequence[Bbox],
    *,
    binary_path: Optional[Path] = None,
    timeout_sec: float = 60.0,
) -> dict:
    """Invoke the Rust ``net-wirelength`` subcommand; return the parsed JSON dict.

    Raises ``BinaryGdsSubprocessError`` if the binary is missing or the
    subprocess fails."""
    payload = {
        "nets": [[[int(p[0]), int(p[1])] for p in net] for net in nets],
        "bboxes": [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in bboxes],
    }
    return invoke_layout_kernel(
        "net-wirelength",
        payload,
        find_binary=_find_binary,
        error_cls=BinaryGdsSubprocessError,
        binary_path=binary_path,
        timeout_sec=timeout_sec,
        missing_binary_hint="  (cd crates/clay_layout && cargo build --release)",
    )


# --- public API (Rust with pure-Python fallback) -----------------------------


def compute_net_wirelength(
    nets: Sequence[Sequence[PinPoint]],
    bboxes: Sequence[Bbox] = (),
    *,
    prefer_rust: bool = True,
    binary_path: Optional[Path] = None,
) -> dict:
    """Compute the full set of net-wirelength metrics.

    Returns a dict with keys ``net_total_span`` (list[int]),
    ``net_max_pin_span`` (list[int]), ``netlist_total_hpwl`` (int) and
    ``overlapping_pairs`` (int). Uses the Rust CLI when available and
    ``prefer_rust`` is True; otherwise falls back to the pure-Python reference.
    The two paths produce identical results.
    """
    if prefer_rust:
        if binary_path is None:
            try:
                from clay._native import load_native

                eng = load_native("_engine_cpp")
                if eng is not None and hasattr(eng, "compute_net_wirelength_cpp"):
                    return eng.compute_net_wirelength_cpp(nets, bboxes)
            except Exception:
                pass
        try:
            return _rust_compute(nets, bboxes, binary_path=binary_path)
        except BinaryGdsSubprocessError:
            pass  # fall through to pure Python
    return {
        "net_total_span": [py_net_total_span(net) for net in nets],
        "net_max_pin_span": [py_net_max_pin_span(net) for net in nets],
        "netlist_total_hpwl": py_netlist_total_hpwl(nets),
        "overlapping_pairs": py_count_overlapping_pairs(list(bboxes)),
    }


def rust_available(binary_path: Optional[Path] = None) -> bool:
    """True if the Rust CLI binary is present on disk (existence only; not version/freshness/executability verified)."""
    return layout_rust_available(_find_binary, binary_path)
