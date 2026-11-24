"""Routing-congestion histogram.

Bins per-channel occupancy values (derived from
:class:`clay.layout.congestion.ChannelOccupancy`) into linear buckets
so a routing-congestion report can be visualized as an ASCII bar chart
or fed to a notebook plot. Read-only over the existing congestion
module; does not mutate ``CongestionReport``.

Occupancy semantics
-------------------
``ChannelOccupancy`` stores an absolute ``wire_count`` per channel. The
histogram converts that to a percentage via ``channel_capacity``:

    occupancy_pct = 100.0 * wire_count / channel_capacity

A channel whose ``wire_count`` equals ``channel_capacity`` is exactly
100% (saturated); anything above counts as overflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .congestion import ChannelOccupancy, CongestionReport


__all__ = [
    "HistogramBin",
    "CongestionHistogram",
    "compute_congestion_histogram",
    "render_histogram_text",
]


@dataclass(frozen=True)
class HistogramBin:
    """One linear bin of the occupancy histogram.

    Attributes:
        lower_pct: bin lower edge, inclusive (percent).
        upper_pct: bin upper edge. Exclusive for all bins except the last,
            where it is inclusive so the ``upper_pct`` boundary is captured.
        count: number of channels whose ``occupancy_pct`` falls in this bin.
    """

    lower_pct: float
    upper_pct: float
    count: int


@dataclass
class CongestionHistogram:
    """Linear occupancy histogram over all channels in a CongestionReport.

    Attributes:
        bins: one :class:`HistogramBin` per bucket, ordered by ``lower_pct``.
        total_channels: total channel count fed into the histogram (sum of
            horizontal_channels + vertical_channels in the source report).
    """

    bins: List[HistogramBin] = field(default_factory=list)
    total_channels: int = 0

    @property
    def overflow_count(self) -> int:
        """Channels with ``occupancy_pct >= 100`` (overcongested)."""
        return sum(b.count for b in self.bins if b.lower_pct >= 100.0)


def _occupancy_pct(channel: ChannelOccupancy, capacity: float) -> float:
    """Convert ``wire_count`` to a percent occupancy."""
    if capacity <= 0:
        raise ValueError(f"channel_capacity must be positive, got {capacity}")
    return 100.0 * channel.wire_count / capacity


def compute_congestion_histogram(
    report: CongestionReport,
    n_bins: int = 10,
    upper_pct: float = 110.0,
    channel_capacity: float = 10.0,
) -> CongestionHistogram:
    """Bucketize channel occupancy percentages into ``n_bins`` linear bins.

    Bins span ``[0.0, upper_pct]`` linearly. Each bin has width
    ``upper_pct / n_bins``. The last bin is closed (inclusive of
    ``upper_pct``); all earlier bins are right-open. Channels whose
    computed ``occupancy_pct`` falls outside ``[0.0, upper_pct]`` are
    clamped into the boundary bin so ``total_channels`` always matches
    the input.

    Args:
        report: a :class:`CongestionReport` to summarize.
        n_bins: number of linear bins. Must be ``>= 1``.
        upper_pct: top of the histogram range (percent). Set above 100 to
            visualize overflow.
        channel_capacity: wires-per-channel that maps to 100% occupancy.

    Returns:
        A :class:`CongestionHistogram` with ``len(bins) == n_bins`` and
        ``total_channels`` equal to the total number of channels in
        ``report``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if upper_pct <= 0:
        raise ValueError(f"upper_pct must be positive, got {upper_pct}")
    if channel_capacity <= 0:
        raise ValueError(
            f"channel_capacity must be positive, got {channel_capacity}"
        )

    bin_width = upper_pct / n_bins
    counts = [0] * n_bins

    channels: List[ChannelOccupancy] = list(report.horizontal_channels) + list(
        report.vertical_channels
    )

    for ch in channels:
        pct = _occupancy_pct(ch, channel_capacity)
        if pct <= 0.0:
            idx = 0
        elif pct >= upper_pct:
            idx = n_bins - 1
        else:
            idx = int(pct / bin_width)
            if idx >= n_bins:
                idx = n_bins - 1
        counts[idx] += 1

    bins = [
        HistogramBin(
            lower_pct=i * bin_width,
            upper_pct=(i + 1) * bin_width,
            count=counts[i],
        )
        for i in range(n_bins)
    ]
    return CongestionHistogram(bins=bins, total_channels=len(channels))


def render_histogram_text(
    hist: CongestionHistogram, max_bar_width: int = 40
) -> str:
    """Render an ASCII bar chart of the histogram, one line per bin.

    Each line has the form ``[lower-upper%] | ####### count``. Bar width
    is scaled by the largest bin so the tallest bin gets ``max_bar_width``
    ``#`` characters; empty histograms render to an empty bar.

    Args:
        hist: a :class:`CongestionHistogram` to render.
        max_bar_width: maximum number of ``#`` characters in any bar.
            Must be ``>= 1``.

    Returns:
        A multi-line string with one bin per line.
    """
    if max_bar_width < 1:
        raise ValueError(
            f"max_bar_width must be >= 1, got {max_bar_width}"
        )

    if not hist.bins:
        return ""

    max_count = max(b.count for b in hist.bins)
    lines: List[str] = []
    for b in hist.bins:
        if max_count > 0:
            bar_len = int(round(max_bar_width * b.count / max_count))
        else:
            bar_len = 0
        bar = "#" * bar_len
        label = f"[{b.lower_pct:6.2f}-{b.upper_pct:6.2f}%]"
        lines.append(f"{label} | {bar} {b.count}")
    return "\n".join(lines)
