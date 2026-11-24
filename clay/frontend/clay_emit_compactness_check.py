"""Two layout-intent obligations for the front-end ``design.clay`` emit + placement.

These name the contract the LDO (and any analog cell) must satisfy once spatial
layout intent is emitted, and they are deliberately written CHECKER-FIRST: both go
RED on the CURRENT engine state BEFORE any fix lands.

    1. ``assert_no_reference_coord_leak`` -- a LEAK AUDIT on the emitted
       ``design.clay`` text. The extracted layout intent must be ABSTRACT RELATIONS
       only (role->region, within-row order, matched-pair adjacency, passive
       banking, region order). It must NEVER pin a device to a RAW REFERENCE
       COORDINATE (an absolute x/y / centroid copied out of the reference layout).
       The audit scans the emit for any absolute-position annotation and goes RED
       if one appears. On TODAY's emit (no spatial intent at all) this is trivially
       clean -- it is calibrated GREEN-on-clean so it can catch a later step that
       injects a coordinate.

    2. ``assert_compact_banking`` -- a COMPACTNESS / BANKING obligation on a
       :class:`clay.layout.generic_placement.Placement`. A RESISTOR (or any passive)
       must be realised as a COMPACT BANK, not a long thin unfolded STRIP. RED if
       any resistor's bbox extent aspect (length/width) exceeds a bound AND its
       length exceeds a folded-bank length bound (so a genuinely small resistor is
       not flagged). The report also carries the placement BBOX + FILL%, so the
       sprawl is measured, not asserted.

Both are INDEPENDENT oracles: the leak audit reads only the emitted TEXT; the
banking checker measures only the REAL placed geometry (device bboxes off the
``Placement``). Neither trusts the emitter's or placer's own bookkeeping. Each
yields a Clay 5-state :class:`Verdict` and goes RED on violation -- never a silent
pass. No per-circuit branch: the banking rule keys off device KIND (resistor /
passive), not a circuit name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Clay 5-state verdict (same vocabulary as clay_emit_check).
PASS = "pass"
FAIL = "fail"
UNMEASURABLE = "unmeasurable"
UNSUPPORTED = "unsupported"
ADVISORY = "advisory"


@dataclass(frozen=True)
class Verdict:
    state: str
    summary: str

    @property
    def is_red(self) -> bool:
        return self.state in (FAIL, UNMEASURABLE, UNSUPPORTED)

    def __bool__(self) -> bool:  # pragma: no cover - guard against truthiness misuse
        raise TypeError(
            "Verdict is 5-state; test .state explicitly, do not use it as a bool"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEAK AUDIT -- no raw reference coordinate in the emitted design.clay
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeakAuditReport:
    verdict: Verdict
    leaks: List[Tuple[str, str]] = field(default_factory=list)  # (annotation, snippet)
    scanned_chars: int = 0


# Annotation / field forms that would pin a device to an ABSOLUTE position copied
# from the reference layout. We forbid the WHOLE family rather than enumerate the
# one syntax a future emitter might pick -- the contract is "no raw coordinate",
# not "no @at". Each pattern requires the coordinate-bearing form (a number, or an
# explicit x/y assignment) so an abstract relation like ``@place_group(g, row=0,
# col=1)`` (rows/cols are ABSTRACT ordinals, not reference dbu) is NOT flagged.
_LEAK_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    # @at(x, y) / @pos(...) / @place_at(...) / @xy(...) / @coord(...) / @centroid(...)
    (
        "absolute-position annotation",
        re.compile(
            r"@(?:at|pos|position|place_at|xy|coord|coordinate|centroid|origin|"
            r"located_at|abs_pos)\s*\(\s*[-+]?\d",
            re.I,
        ),
    ),
    # x = <num> / y = <num> / cx = / cy = as a standalone position field (NOT W=/L=,
    # those are sizes; the alias list is exhaustive position-field stems).
    (
        "absolute x/y position field",
        re.compile(
            r"(?<![A-Za-z_])(?:abs_?x|abs_?y|pos_?x|pos_?y|cx|cy|"
            r"origin_?x|origin_?y|center_?x|center_?y|centroid_?x|centroid_?y)\s*=\s*"
            r"[-+]?\d",
            re.I,
        ),
    ),
    # a bare ``@symmetric`` axis literal carrying a numeric axis position (an axis
    # at a reference dbu, not the abstract ``vertical`` / ``horizontal`` literal).
    (
        "numeric symmetry-axis position",
        re.compile(r"@symmetric\s*\(\s*[-+]?\d", re.I),
    ),
)


def assert_no_reference_coord_leak(design_clay: str) -> LeakAuditReport:
    """RED if the emitted ``design.clay`` pins any device to a RAW reference
    coordinate (an absolute x/y / centroid / numeric axis copied from the reference
    layout). PASS (calibrated GREEN-on-clean) when the emit carries only abstract
    relations (region/row assignment, ordering, banking, matched-pair adjacency).

    UNMEASURABLE only on an empty input (nothing to audit).
    """
    text = design_clay or ""
    if not text.strip():
        return LeakAuditReport(
            verdict=Verdict(UNMEASURABLE, "empty design.clay -- nothing to audit"),
            scanned_chars=0,
        )

    leaks: List[Tuple[str, str]] = []
    for name, pat in _LEAK_PATTERNS:
        for m in pat.finditer(text):
            start = max(0, m.start() - 16)
            end = min(len(text), m.end() + 16)
            snippet = text[start:end].replace("\n", " ").strip()
            leaks.append((name, snippet))

    if leaks:
        kinds = sorted({k for k, _ in leaks})
        verdict = Verdict(
            FAIL,
            f"COORDINATE LEAK: {len(leaks)} raw-position annotation(s) in the "
            f"emitted design.clay ({', '.join(kinds)}); layout intent must be "
            f"abstract relations only, never a raw reference coordinate.",
        )
    else:
        verdict = Verdict(
            PASS,
            "leak-free: emitted design.clay carries no raw reference coordinate "
            f"(scanned {len(text)} chars; only abstract relations permitted).",
        )
    return LeakAuditReport(verdict=verdict, leaks=leaks, scanned_chars=len(text))


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPACTNESS / BANKING -- a resistor must be a compact bank, not a strip
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompactnessReport:
    verdict: Verdict
    bbox_um: Tuple[float, float] = (0.0, 0.0)
    fill_pct: float = 0.0
    # one (label, length_um, width_um, aspect) per RED resistor strip.
    strip_resistors: List[Tuple[str, float, float, float]] = field(default_factory=list)
    # one (label, length_um, width_um, aspect) for EVERY resistor (evidence).
    all_resistors: List[Tuple[str, float, float, float]] = field(default_factory=list)
    n_devices: int = 0


# A poly resistor folded into a compact serpentine BANK has a small bbox with a
# near-square aspect. An UNFOLDED single-segment strip is hundreds of um long and
# ~1.6 um wide -> an aspect in the tens-to-hundreds. The bound pair below RED-flags
# a strip while leaving a genuinely small resistor (a single short unit) green.
_MAX_RESISTOR_ASPECT = 10.0      # length/width beyond this == a strip, not a bank
_MAX_BANK_LENGTH_UM = 50.0       # a folded bank's longest extent stays well under


def _resistor_kind(kind: str) -> bool:
    k = (kind or "").strip().lower()
    return k in ("res", "resistor", "r", "ppolyf_u") or k.startswith("res")


def assert_compact_banking(
    placement: object,
    *,
    max_aspect: float = _MAX_RESISTOR_ASPECT,
    max_bank_length_um: float = _MAX_BANK_LENGTH_UM,
) -> CompactnessReport:
    """RED if any RESISTOR on ``placement`` is realised as a long thin STRIP
    (aspect > ``max_aspect`` AND longest extent > ``max_bank_length_um``) instead
    of a compact folded bank. Reports the placement bbox (um) + fill%.

    UNMEASURABLE if the placement has no devices.
    """
    devices = getattr(placement, "devices", None)
    if not devices:
        return CompactnessReport(
            verdict=Verdict(UNMEASURABLE, "placement has no devices -- nothing to measure"),
        )

    xs0: List[int] = []
    ys0: List[int] = []
    xs1: List[int] = []
    ys1: List[int] = []
    dev_area = 0
    all_res: List[Tuple[str, float, float, float]] = []
    strips: List[Tuple[str, float, float, float]] = []

    for label, pd in devices.items():
        bbox = getattr(pd, "bbox", None)
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
        dev_area += max(0, x1 - x0) * max(0, y1 - y0)
        if _resistor_kind(getattr(pd, "kind", "")):
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            length = max(w, h)
            width = max(min(w, h), 1)
            aspect = length / width
            entry = (str(label), length / 1000.0, width / 1000.0, aspect)
            all_res.append(entry)
            if aspect > max_aspect and (length / 1000.0) > max_bank_length_um:
                strips.append(entry)

    if not xs0:
        return CompactnessReport(
            verdict=Verdict(UNMEASURABLE, "no device bboxes -- nothing to measure"),
            n_devices=len(devices),
        )

    bx0, by0, bx1, by1 = min(xs0), min(ys0), max(xs1), max(ys1)
    w_um = (bx1 - bx0) / 1000.0
    h_um = (by1 - by0) / 1000.0
    bbox_area = (bx1 - bx0) * (by1 - by0)
    fill = 100.0 * dev_area / bbox_area if bbox_area else 0.0

    all_res.sort(key=lambda t: -t[1])
    strips.sort(key=lambda t: -t[1])

    if strips:
        worst = strips[0]
        verdict = Verdict(
            FAIL,
            f"NOT BANKED: {len(strips)} of {len(all_res)} resistor(s) realised as "
            f"long thin strips (worst {worst[0]}: {worst[1]:.1f}x{worst[2]:.1f} um, "
            f"aspect {worst[3]:.0f}) instead of a compact bank; placement bbox "
            f"{w_um:.1f}x{h_um:.1f} um, fill {fill:.2f}%.",
        )
    else:
        verdict = Verdict(
            PASS,
            f"banked: all {len(all_res)} resistor(s) compact (aspect <= {max_aspect:g} "
            f"or length <= {max_bank_length_um:g} um); placement bbox "
            f"{w_um:.1f}x{h_um:.1f} um, fill {fill:.2f}%.",
        )

    return CompactnessReport(
        verdict=verdict,
        bbox_um=(round(w_um, 1), round(h_um, 1)),
        fill_pct=round(fill, 2),
        strip_resistors=strips,
        all_resistors=all_res,
        n_devices=len(devices),
    )


__all__ = [
    "Verdict",
    "LeakAuditReport",
    "CompactnessReport",
    "assert_no_reference_coord_leak",
    "assert_compact_banking",
]
