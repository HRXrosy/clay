"""Matching-distance metric for Matching annotation pairs.

This module walks all ``Matching`` annotations attached to an
``AnnotatedCell``, pairs each device with its declared partner, and
computes simple per-pair ratio signatures over the three sizing
co-variables that matter for layout matching: W, L, and finger_count.

The output is a small report containing one signature per unordered
pair (bidirectional declarations such as ``M_in_p`` matched to
``M_in_n`` and ``M_in_n`` matched to ``M_in_p`` are deduplicated to
exactly one canonical entry, ordered alphabetically by device name).

A separate helper ``compute_l1_distance`` reduces a signature to a
single non-negative scalar by summing ``|1 - ratio|`` across the three
ratios. A perfect match yields 0.0.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .annotations import AnnotatedCell, Matching


@dataclass(frozen=True)
class MatchingPairSignature:
    """Per-pair ratio signature for a single Matching declaration.

    Attributes
    ----------
    device_a, device_b:
        Canonical alphabetical ordering of the two paired devices
        (``device_a < device_b`` always).
    w_ratio:
        Width ratio ``W_a / W_b``. Exactly ``1.0`` when both widths are
        identical.
    l_ratio:
        Length ratio ``L_a / L_b``.
    finger_count_ratio:
        ``finger_count_a / finger_count_b``.
    kind:
        Name of the ``MatchingKind`` enum value (e.g. ``"RANDOM_VT"``).
    is_exact_match:
        ``True`` iff all three ratios are exactly ``1.0``.
    """

    device_a: str
    device_b: str
    w_ratio: float
    l_ratio: float
    finger_count_ratio: float
    kind: str
    is_exact_match: bool


@dataclass
class MatchingMetricReport:
    """Aggregate report across all Matching pairs in a cell."""

    total_pairs: int
    exact_match_pairs: int
    pairs: List[MatchingPairSignature] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        if not self.total_pairs:
            return 0.0
        return self.exact_match_pairs / self.total_pairs


def _safe_ratio(a: float, b: float) -> float:
    """Return ``a / b`` with the convention that identical values map
    to exactly ``1.0`` (avoids floating-point drift like
    ``0.5 / 0.5 != 1.0`` on some platforms) and that a zero divisor
    is treated as ``1.0`` when the numerator is also zero, otherwise
    raises ``ZeroDivisionError``.
    """
    if a == b:
        return 1.0
    if b == 0:
        if a == 0:
            return 1.0
        raise ZeroDivisionError(
            f"matching ratio divisor is zero (numerator={a})"
        )
    return a / b


def extract_matching_pairs(
    cell: AnnotatedCell,
    sizing: Dict[str, Tuple[float, float]],
) -> MatchingMetricReport:
    """Walk all ``Matching`` annotations in ``cell`` and compute per-pair
    ratio signatures.

    Parameters
    ----------
    cell:
        Source ``AnnotatedCell``. Its ``annotations`` map provides the
        Matching declarations; its ``layouts`` map provides each
        device's ``finger_count``.
    sizing:
        ``device_name -> (W_um, L_um)`` lookup. Devices referenced by a
        Matching annotation must appear here, else ``KeyError`` is
        raised (so callers cannot silently produce ratios over missing
        devices).

    Returns
    -------
    MatchingMetricReport
        Aggregated, deduplicated, alphabetically-canonical view of the
        cell's Matching pairs.
    """
    seen: Dict[Tuple[str, str], MatchingPairSignature] = {}

    for device, anns in cell.annotations.items():
        for ann in anns:
            if not isinstance(ann, Matching):
                continue

            other = ann.other
            # Canonical ordering: device_a < device_b alphabetically.
            if device == other:
                # A self-match is degenerate; skip rather than emit a
                # trivial 1.0/1.0/1.0 entry that pollutes the rate.
                continue
            if device < other:
                a, b = device, other
            else:
                a, b = other, device

            key = (a, b)
            if key in seen:
                continue  # bidirectional duplicate

            if a not in sizing:
                raise KeyError(
                    f"sizing missing for device {a!r} referenced by Matching"
                )
            if b not in sizing:
                raise KeyError(
                    f"sizing missing for device {b!r} referenced by Matching"
                )

            w_a, l_a = sizing[a]
            w_b, l_b = sizing[b]

            fc_a = (
                cell.layouts[a].finger_count if a in cell.layouts else 1
            )
            fc_b = (
                cell.layouts[b].finger_count if b in cell.layouts else 1
            )

            w_ratio = _safe_ratio(float(w_a), float(w_b))
            l_ratio = _safe_ratio(float(l_a), float(l_b))
            fc_ratio = _safe_ratio(float(fc_a), float(fc_b))

            is_exact = (
                w_ratio == 1.0 and l_ratio == 1.0 and fc_ratio == 1.0
            )

            sig = MatchingPairSignature(
                device_a=a,
                device_b=b,
                w_ratio=w_ratio,
                l_ratio=l_ratio,
                finger_count_ratio=fc_ratio,
                kind=ann.kind.name,
                is_exact_match=is_exact,
            )
            seen[key] = sig

    pairs = [seen[k] for k in sorted(seen.keys())]
    exact_count = sum(1 for s in pairs if s.is_exact_match)

    return MatchingMetricReport(
        total_pairs=len(pairs),
        exact_match_pairs=exact_count,
        pairs=pairs,
    )


def compute_l1_distance(sig: MatchingPairSignature) -> float:
    """Return ``|1 - w_ratio| + |1 - l_ratio| + |1 - finger_count_ratio|``.

    A perfectly matched pair yields ``0.0``. The result is always
    non-negative.
    """
    return (
        abs(1.0 - sig.w_ratio)
        + abs(1.0 - sig.l_ratio)
        + abs(1.0 - sig.finger_count_ratio)
    )
