"""Common-centroid placement DIAGNOSTIC (layout-intent methodology point 7).

The layout-intent contract's matched-device methodology says matched devices
should be INTERDIGITATED / COMMON-CENTROID so that a process gradient (oxide
thickness, doping, lithographic, stress) cancels out of the matched pair. This
module reports WHERE a placed cell stands against that ideal. It is a
DIAGNOSTIC, not a hard pass/fail obligation: it surfaces the distinction between
the two gradient-cancellation regimes without forcing a failure on a layout that
is only mirror-symmetric.

This is a separate concern from the finger-interleaving TRANSFORM in
:mod:`clay.layout.common_centroid` (which re-orders a device's fingers into a
shared column). That module BUILDS an interdigitated column; this module
INSPECTS an already-placed cell and classifies each matched device pair against
the common-centroid ideal.

Mirror vs common-centroid (the honest distinction)
==================================================
Consider a matched device pair ``(A, A')`` and a process parameter that varies
LINEARLY across the cell.

  * MIRROR placement puts ``A`` at ``+x`` and its match ``A'`` at the equal-and-
    opposite ``-x``. The PAIR centroid (the midpoint of the two device centroids)
    then lands ON the axis, so the linear-ACROSS-the-axis component of the
    gradient is equal-and-opposite at the two devices and cancels from the pair
    average. But each device's OWN centroid is still offset from the axis (``A``
    sits entirely at ``+x``, ``A'`` entirely at ``-x``), so a higher-order
    (curvature) term, or a gradient along a direction the mirror does not
    straddle, does NOT cancel the way an interdigitated layout makes it cancel.

  * TRUE COMMON-CENTROID placement interleaves the two devices' sub-elements
    (ABBA / ABAB) so that EACH device's own element centroid COINCIDES with the
    other's (and with the pair centroid). Then both devices see the SAME average
    of the gradient field over their elements, so the linear gradient cancels in
    EVERY direction, not only across one mirror axis, and the pair is far more
    robust to the higher-order terms a single mirror leaves behind.

The current comparator and OTA cells in this repo use MIRROR symmetry: the
matched device pair's centroid lands on the axis (``pair_centroid_offset`` ~ 0),
but the two devices are NOT interdigitated, so their individual centroids are far
apart. This diagnostic reports exactly that, and is the basis for a FUTURE
interdigitation pass that would turn a "mirror" pair into a "common_centroid"
one.

Device <-> diff-pair mapping
============================
A diff pair is a NET pair (e.g. ``(intern, interp)``); it is physically realised
by one or more matched DEVICE pairs. The mapping from a net diff pair to its
matched device pair is the SAME machinery the placement-symmetry checker uses --
this module reuses :func:`clay.layout.placement_symmetry.placement_symmetry_obligations`
to enumerate the matched device pairs (the router's ``_PAIR_LEFT`` / ``_PAIR_RIGHT``
annotation, or the symmetric-naming fallback) rather than duplicating it. The
placement-symmetry obligation already carries the device pair and is computed on
the SAME cell axis; this diagnostic adds the common-centroid classification on
top of the SAME device centroids.

Relative quantities only
========================
Per the project-wide consistency rule, every reported quantity is RELATIVE (a
fraction of the cell extent), never an absolute nm, so the diagnostic is process-
and scale-neutral (scaling every coordinate leaves it unchanged):

  * ``pair_centroid_offset`` -- |pair-centroid x - axis| / cell_width. Small means
    the pair centroid sits on the axis (a mirror pair already achieves this).
  * ``centroid_coincidence`` -- the distance between the two devices' OWN
    centroids, divided by the cell DIAGONAL. ~0 means the two centroids coincide
    (interdigitated / common-centroid); large means each device sits on its own
    side (mirror).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.placement_symmetry import (
    device_centroids,
    placement_symmetry_obligations,
)

# A pair is COMMON-CENTROID when the two devices' own centroids coincide to within
# this RELATIVE tolerance (a fraction of the cell diagonal). A genuine
# interdigitated layout drives this to ~0; a mirror layout leaves it at the
# pair's half-separation, far above the tolerance.
_DEFAULT_COINCIDENCE_TOL = 0.02

# A pair is balanced ABOUT the axis (the mirror property) when its pair-centroid
# offset is within this RELATIVE tolerance (a fraction of the cell width). Both a
# mirror pair and a common-centroid pair satisfy this; an asymmetric placement
# does not.
_DEFAULT_AXIS_TOL = 0.02

_MIRROR_NOTE = (
    "mirror: pair centroid on the axis -> cancels the linear gradient component "
    "ACROSS the axis, but each device's own centroid is offset, so it does NOT "
    "cancel gradients along other directions / higher-order terms"
)
_COMMON_CENTROID_NOTE = (
    "common_centroid: the two devices' centroids coincide (interdigitated) -> "
    "cancels the linear gradient in EVERY direction, more robust than mirror to "
    "higher-order (curvature) terms"
)
_ASYMMETRIC_NOTE = (
    "asymmetric: the pair centroid is off the axis AND the device centroids do "
    "not coincide -> neither the mirror nor the common-centroid gradient "
    "cancellation holds"
)


def _cell_extent(centroids: Dict[str, Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """(min_x, min_y, max_x, max_y) over every device centroid."""
    xs = [c[0] for c in centroids.values()]
    ys = [c[1] for c in centroids.values()]
    return (min(xs), min(ys), max(xs), max(ys))


def _classify(
    pair_centroid_offset: float,
    centroid_coincidence: float,
    coincidence_tol: float,
    axis_tol: float,
) -> Tuple[str, str]:
    """Classify a matched pair's placement and return ``(placement_kind, note)``.

    Rule:
      * ``common_centroid`` -- the two devices' OWN centroids coincide
        (``centroid_coincidence <= coincidence_tol``): interdigitated, the
        strongest gradient cancellation. (A common-centroid pair's centroid is on
        the axis too, so this case wins regardless of the axis offset.)
      * ``mirror`` -- the centroids do NOT coincide but the PAIR centroid sits on
        the axis (``pair_centroid_offset <= axis_tol``): equidistant-opposite
        about the axis, cancels the linear-across-axis component only.
      * ``asymmetric`` -- neither holds.
    """
    if centroid_coincidence <= coincidence_tol:
        return "common_centroid", _COMMON_CENTROID_NOTE
    if pair_centroid_offset <= axis_tol:
        return "mirror", _MIRROR_NOTE
    return "asymmetric", _ASYMMETRIC_NOTE


def common_centroid_diagnostic(
    routed_cell,
    intent,
    *,
    axis: Optional[float] = None,
    coincidence_tol: float = _DEFAULT_COINCIDENCE_TOL,
    axis_tol: float = _DEFAULT_AXIS_TOL,
) -> List[Dict[str, object]]:
    """Per matched device pair, classify its placement against the common-centroid ideal.

    For every matched DEVICE pair the placement-symmetry checker maps each
    ``intent.diff_pairs`` net pair to (reusing that checker's device<->net
    mapping, NOT duplicating it), report:

      * ``pair_centroid_offset`` -- |pair-centroid x - axis| / cell_width
        (RELATIVE). Small means the PAIR centroid is on the axis -- the mirror
        property; a mirror placement is enough to achieve this.
      * ``centroid_coincidence`` -- the distance between the two devices' OWN
        centroids / cell diagonal (RELATIVE). ~0 means the devices are
        interdigitated (common-centroid); large means each device sits on its own
        side of the axis (mirror).
      * ``placement_kind`` -- ``"common_centroid"`` (centroids coincide within
        ``coincidence_tol``), ``"mirror"`` (centroids do not coincide but the pair
        centroid is on the axis within ``axis_tol``), else ``"asymmetric"``.
      * ``gradient_cancel`` -- a note on what each placement_kind cancels (mirror:
        linear-across-axis only; common-centroid: linear in every direction +
        more robust to curvature terms).

    ``axis`` defaults to the mean x of every device centroid (a symmetric cell's
    axis); a caller may pass an explicit axis. All reported numbers are RELATIVE
    fractions, so the diagnostic is scale- and process-neutral.

    This is a DIAGNOSTIC: it never raises and never forces a failure. It surfaces
    where the layout stands (the comparator/OTA today are ``"mirror"``), which is
    the basis for a future interdigitation pass.

    Returns one plain dict per matched device pair (empty if the cell exposes no
    device centroids, or the intent declares no diff pairs).
    """
    centroids = device_centroids(routed_cell)
    if not centroids:
        return []
    x0, y0, x1, y1 = _cell_extent(centroids)
    cell_width = max(1, x1 - x0)
    cell_height = max(1, y1 - y0)
    cell_diag = max(1.0, math.hypot(cell_width, cell_height))
    if axis is None:
        axis = sum(c[0] for c in centroids.values()) / len(centroids)

    # Reuse the placement-symmetry checker to enumerate the matched DEVICE pairs of
    # every diff pair (its router/plan device<->net mapping is the single source of
    # truth -- this diagnostic does not re-derive it). It runs on the SAME axis so
    # the two checkers agree on the axis a pair is balanced about.
    sym_obs = placement_symmetry_obligations(routed_cell, intent, axis=axis)

    out: List[Dict[str, object]] = []
    for so in sym_obs:
        left, right = so["devices"]
        if left not in centroids or right not in centroids:
            continue
        (lx, ly) = centroids[left]
        (rx, ry) = centroids[right]
        pair_cx = (lx + rx) / 2.0
        pair_cy = (ly + ry) / 2.0
        # The mirror property is about the AXIS (the cell's x symmetry axis): the
        # pair centroid's x should land on the axis. (The pair-centroid y is not an
        # axis quantity -- a matched pair sits at one height by construction.)
        pair_centroid_offset = abs(pair_cx - axis) / cell_width
        centroid_coincidence = math.hypot(lx - rx, ly - ry) / cell_diag
        placement_kind, note = _classify(
            pair_centroid_offset, centroid_coincidence, coincidence_tol, axis_tol
        )
        out.append({
            "pair": so["pair"],
            "devices": (left, right),
            "pair_centroid_offset": pair_centroid_offset,
            "centroid_coincidence": centroid_coincidence,
            "pair_centroid": (pair_cx, pair_cy),
            "placement_kind": placement_kind,
            "gradient_cancel": note,
            "coincidence_tol": coincidence_tol,
            "axis_tol": axis_tol,
            "match_kind": so.get("match_kind", "differential"),
        })
    return out


def common_centroid_summary(diagnostics: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-pair common-centroid diagnostics into a summary.

    Returns ``{n_pairs, n_common_centroid, n_mirror, n_asymmetric,
    worst_pair_offset}``:

      * ``n_pairs`` -- total matched device pairs diagnosed;
      * ``n_common_centroid`` / ``n_mirror`` / ``n_asymmetric`` -- how many fall
        into each ``placement_kind`` (these three sum to ``n_pairs``);
      * ``worst_pair_offset`` -- the largest ``pair_centroid_offset`` across all
        pairs (the pair whose centroid is farthest off the axis); 0.0 for an empty
        diagnostic.

    Honest: a high ``n_mirror`` with a near-zero ``worst_pair_offset`` is the
    EXPECTED state of the current comparator/OTA -- the pairs are balanced about
    the axis (mirror) but not interdigitated (common-centroid). This is a
    diagnostic summary, not a pass/fail verdict.
    """
    n_common = sum(1 for d in diagnostics if d.get("placement_kind") == "common_centroid")
    n_mirror = sum(1 for d in diagnostics if d.get("placement_kind") == "mirror")
    n_asym = sum(1 for d in diagnostics if d.get("placement_kind") == "asymmetric")
    worst = max((float(d["pair_centroid_offset"]) for d in diagnostics), default=0.0)
    return {
        "n_pairs": len(diagnostics),
        "n_common_centroid": n_common,
        "n_mirror": n_mirror,
        "n_asymmetric": n_asym,
        "worst_pair_offset": worst,
    }


__all__ = [
    "common_centroid_diagnostic",
    "common_centroid_summary",
]
