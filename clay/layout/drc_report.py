"""Backend-neutral DRC violation report reader.

Parses a signoff DRC tool's violation database into a flat, tool-independent
``RuleViolation`` list so the error-intent solver
(:mod:`clay.layout.fault_intent`) can JOIN deck-level violations to the routed
geometry's net/decision provenance -- the same way the PEX layer is backend-neutral
(:mod:`clay.layout.pex`). This is the GENERAL capability: any circuit's real DRC
output becomes decision-space-diagnosable; a special circuit (fc_ota) only
validates the strategy, it is never special-cased.

Currently implements the KLayout RDB (``.lyrdb``) reader (the gf180 signoff
backend). A Magic / external_drc reader would add a sibling parser returning the same
``RuleViolation`` list; the solver never sees the tool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Bbox = Tuple[int, int, int, int]

# KLayout RDB geometry values are in MICRONS; layout/router geometry is in nm.
_UM_TO_NM = 1000.0
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_ITEM = re.compile(r"<item>(.*?)</item>", re.S)
_CATEGORY = re.compile(r"<category>\s*'?([^'<]+?)'?\s*</category>", re.S)
_VALUE = re.compile(r"<value>(.*?)</value>", re.S)


@dataclass(frozen=True)
class RuleViolation:
    """One DRC violation, tool-independent.

    ``rule`` is the deck rule name (e.g. ``M1.2a``); ``bbox`` is the violation's
    bounding box in nm; ``polygon`` is the raw vertex list in nm (for a precise
    locus). The error-intent solver attributes the violation to the net(s) /
    decision(s) whose routed geometry coincides with ``bbox``."""

    rule: str
    bbox: Bbox
    polygon: Tuple[Tuple[int, int], ...]

    @property
    def locus(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) // 2, (y0 + y1) // 2)


def _coords_nm(value_text: str) -> List[Tuple[int, int]]:
    """Every (x, y) vertex in a KLayout RDB value string, converted um -> nm.

    Handles ``polygon: (x,y;...)``, ``box: (x,y;x,y)``, ``edge: (x,y;x,y)`` and
    ``edge-pair: (...)/(...)`` uniformly by pairing all numbers in order."""
    nums = [float(t) for t in _NUM.findall(value_text)]
    pts: List[Tuple[int, int]] = []
    for i in range(0, len(nums) - 1, 2):
        pts.append((round(nums[i] * _UM_TO_NM), round(nums[i + 1] * _UM_TO_NM)))
    return pts


def parse_lyrdb(path: str | Path) -> List[RuleViolation]:
    """Parse a KLayout ``.lyrdb`` DRC report into a ``RuleViolation`` list (nm).

    Each ``<item>`` is one violation: its ``<category>`` is the rule name and its
    ``<value>`` entries carry the geometry (in um). An item with no parseable
    geometry is skipped. Returns [] for a clean (item-free) report."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out: List[RuleViolation] = []
    for body in _ITEM.findall(txt):
        mcat = _CATEGORY.search(body)
        rule = mcat.group(1).strip() if mcat else "?"
        pts: List[Tuple[int, int]] = []
        for v in _VALUE.findall(body):
            pts.extend(_coords_nm(v))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append(RuleViolation(
            rule=rule,
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            polygon=tuple(pts),
        ))
    return out


def violation_counts(violations: List[RuleViolation]) -> dict:
    """``{rule: count}`` summary (for a quick before/after headline)."""
    out: dict = {}
    for v in violations:
        out[v.rule] = out.get(v.rule, 0) + 1
    return out


__all__ = ["RuleViolation", "parse_lyrdb", "violation_counts"]
