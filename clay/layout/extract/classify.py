"""PHASE-B PERTURBATION CLASSIFIER -- label each residual coordinate DON'T-CARE
vs MISSING-PRIMITIVE.

WHAT THIS IS
============
The no-data-leakage degradation experiment extracts DECLARATIVE layout intent from
a reference design. Starting from the full reference coordinates, it asks, for each
device's position: is the position FORCED by the design (some relation pins it -> we
are MISSING a primitive that must be discovered in Phase C) or is it a FREE choice
(a DON'T-CARE = genuine design freedom)? The discriminator is PERTURBATION:

  move the device away from its reference position
    -> the layout STAYS valid under the move  -> DON'T-CARE (position is free)
    -> only the original position is valid     -> MISSING PRIMITIVE (a relation
                                                  pins it; Phase C must discover it)

The validity test each move runs against is the shared oracle
:func:`clay.layout.extract.validity.is_valid` (overlap-free + DRC-spacing-legal +
declared-relations-hold), with the reference passed as ``baseline_positions`` so the
symmetry sub-check is baseline-gated (only pairs that mirror in the reference are
enforced) -- see ``validity.py``.

LEAKAGE DISCIPLINE (the whole scientific point)
===============================================
Perturbing FROM the reference position to TEST whether it is forced is the
degradation-start analysis and is allowed. But the OUTPUT of classification is a
LABEL (``'dont_care'`` | ``'missing_primitive'``) plus which-axis and the
perturbation evidence -- NEVER a stored reference coordinate. A DON'T-CARE device
means: declare its position FREE (no value stored). A MISSING-PRIMITIVE device means:
some relation must be discovered to pin it (Phase C), still with no raw coordinate
persisted. ``classify_*`` never returns or stores a reference coordinate; the moved
positions it builds are transient inputs to the oracle.

BASELINE-RELATIVE VALIDITY
==========================
The reference layout is not necessarily GLOBALLY valid: a folded RF reference can
carry a documented interdigitation nesting (e.g. a band_only diode-reference whose
inferred bbox nests inside its mirror partner's finger bank). So a perturbation
"STAYED valid" is judged RELATIVE TO THE BASELINE: a move counts as staying valid iff
it introduces NO NEW violation that INVOLVES THE MOVED DEVICE relative to the
reference's own violation set. Pre-existing violations the move does not touch (and
violations on other devices) do not, by themselves, flip a device to forced. This is
exactly :func:`clay.layout.extract.validity.validity_delta`'s ``newly_violated``
restricted to the moved device.

UNITS: every coordinate / size is in MICRONS (the shared ``device_bboxes`` contract).
``positions`` are device CENTROIDS ``(cx, cy)``; ``sizes`` are footprints ``(w, h)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .validity import (
    DEFAULT_MIN_GAP_UM,
    is_valid,
    validity_delta,
)

_Pos = Tuple[float, float]
_Size = Tuple[float, float]
_Bbox = Tuple[float, float, float, float]
_Delta = Tuple[float, float]

# Default perturbation magnitudes (microns). Each is applied as +/- in x, +/- in y,
# and on the four diagonals, so a device is probed at 8 displacements per magnitude.
DEFAULT_DELTAS_UM: Tuple[float, ...] = (5.0, 10.0, 20.0)

# Decision thresholds on the fraction of perturbations that STAYED valid.
#   >= DONT_CARE_FRACTION of probes stay valid -> the position is FREE -> dont_care.
#   <= FORCED_MAX_VALID probes stay valid       -> the position is pinned -> missing_primitive.
# Probes whose fraction lands strictly between the two thresholds are leaning-forced
# (a relation constrains the device on at least one axis but not in every direction);
# we label them ``missing_primitive`` (something must be discovered to pin them) and
# record the partial-freedom evidence. Named constants so the rule is auditable.
DONT_CARE_FRACTION: float = 0.50  # a MAJORITY of probes staying valid -> free
FORCED_MAX_VALID: int = 1         # at most this many probes valid -> forced


# --------------------------------------------------------------------------- #
# result type                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class PerturbResult:
    """Classification of ONE device's reference position.

    ``label`` is the verdict: ``'dont_care'`` (position is free -> declare it free,
    store no coordinate) or ``'missing_primitive'`` (position is pinned by some
    relation Phase C must discover). ``axis`` records WHERE the freedom / constraint
    lives: ``'both'`` (free or forced in x and y alike), ``'x'`` (only x-moves stay
    valid -> y is the constrained axis) or ``'y'`` (only y-moves stay valid).
    ``n_valid_perturbations`` / ``n_total_perturbations`` are the raw probe tallies;
    ``deltas_that_stayed_valid`` lists the ``(dx, dy)`` displacements that kept the
    layout valid (evidence, NOT a stored position); ``evidence`` is a human string.
    """

    device: str
    axis: str  # 'x' | 'y' | 'both'
    label: str  # 'dont_care' | 'missing_primitive'
    n_valid_perturbations: int
    n_total_perturbations: int
    deltas_that_stayed_valid: List[_Delta] = field(default_factory=list)
    evidence: str = ""


# --------------------------------------------------------------------------- #
# reference adapters (centroids + sizes from device_bboxes)                    #
# --------------------------------------------------------------------------- #


def reference_positions(intent: Any) -> Dict[str, _Pos]:
    """Centroids ``{name -> (cx, cy)}`` from ``intent.device_bboxes`` (microns).

    These are the reference positions the classifier perturbs FROM (the
    degradation-start). They are transient classifier inputs -- the classifier
    OUTPUT never persists them (leakage discipline).
    """
    out: Dict[str, _Pos] = {}
    for name, bb in (getattr(intent, "device_bboxes", None) or {}).items():
        x0, y0, x1, y1 = (float(v) for v in bb)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        out[str(name)] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return out


def device_sizes(intent: Any) -> Dict[str, _Size]:
    """Footprints ``{name -> (w, h)}`` from ``intent.device_bboxes`` (microns)."""
    out: Dict[str, _Size] = {}
    for name, bb in (getattr(intent, "device_bboxes", None) or {}).items():
        x0, y0, x1, y1 = (float(v) for v in bb)
        out[str(name)] = (abs(x1 - x0), abs(y1 - y0))
    return out


# --------------------------------------------------------------------------- #
# the perturbation probe set                                                  #
# --------------------------------------------------------------------------- #


def _displacements(deltas_um: Tuple[float, ...]) -> List[_Delta]:
    """The ``(dx, dy)`` displacements probed for each device.

    For each magnitude d: +/-d in x (2), +/-d in y (2), and the four diagonals
    (+-d, +-d) (4) -> 8 probes per magnitude. Deterministic order (no randomness),
    so a classification is byte-reproducible.
    """
    out: List[_Delta] = []
    for d in deltas_um:
        out.extend(
            [
                (+d, 0.0),
                (-d, 0.0),
                (0.0, +d),
                (0.0, -d),
                (+d, +d),
                (+d, -d),
                (-d, +d),
                (-d, -d),
            ]
        )
    return out


def _stayed_valid(
    positions: Dict[str, _Pos],
    moved: Dict[str, _Pos],
    sizes: Dict[str, _Size],
    intent: Any,
    device: str,
    *,
    min_gap_um: float,
) -> bool:
    """Did moving ``device`` keep the layout valid RELATIVE TO THE BASELINE?

    True iff the perturbation introduces NO NEW violation that involves ``device``
    relative to the reference's own violation set (baseline-relative validity, see
    module docstring). Pre-existing violations the move does not touch -- and
    violations on other devices -- do not flip ``device`` to forced.
    """
    d = validity_delta(
        positions,
        moved,
        sizes,
        intent,
        min_gap_um=min_gap_um,
        baseline_positions=positions,
    )
    for _kind, devs in d.newly_violated:
        if device in devs:
            return False
    return True


# --------------------------------------------------------------------------- #
# classify one device                                                         #
# --------------------------------------------------------------------------- #


def classify_device(
    positions: Dict[str, _Pos],
    sizes: Dict[str, _Size],
    intent: Any,
    device: str,
    *,
    deltas_um: Tuple[float, ...] = DEFAULT_DELTAS_UM,
    min_gap_um: float = DEFAULT_MIN_GAP_UM,
) -> PerturbResult:
    """Classify ``device``'s reference position as DON'T-CARE vs MISSING-PRIMITIVE.

    Perturbs ``device`` from ``positions[device]`` (pass the reference centroids in)
    by each displacement in :func:`_displacements` -- only ``device`` moves, every
    other device stays put -- and asks the oracle whether each move kept the layout
    valid relative to the baseline. The reference is passed as
    ``baseline_positions`` so the symmetry sub-check is baseline-gated.

    DECISION RULE (thresholds are the named constants
    :data:`DONT_CARE_FRACTION` / :data:`FORCED_MAX_VALID`):

      * a MAJORITY (``>= DONT_CARE_FRACTION``) of probes stay valid -> ``'dont_care'``
        (the position is free design space; declare it free, store no coordinate);
      * at most :data:`FORCED_MAX_VALID` probes stay valid -> ``'missing_primitive'``
        (only the original position is valid -> a relation pins it; Phase C must
        discover it);
      * in between -> leaning forced -> ``'missing_primitive'`` with partial-freedom
        evidence (constrained on at least one axis / direction).

    ``axis`` summarises WHERE the freedom or constraint lives: ``'both'`` when x- and
    y-moves behave alike, else the axis whose moves dominated the surviving probes.

    Returns a :class:`PerturbResult`. NEVER stores a reference coordinate -- the
    moved positions are transient oracle inputs (leakage discipline).
    """
    probes = _displacements(deltas_um)
    base = positions[device]
    cx, cy = base[0], base[1]

    stayed: List[_Delta] = []
    n_x_valid = n_x_total = 0
    n_y_valid = n_y_total = 0
    for dx, dy in probes:
        moved = dict(positions)
        moved[device] = (cx + dx, cy + dy)
        ok = _stayed_valid(
            positions, moved, sizes, intent, device, min_gap_um=min_gap_um
        )
        # axis bookkeeping on the pure-axis probes (diagonals count for both axes).
        moves_x = dx != 0.0
        moves_y = dy != 0.0
        if moves_x:
            n_x_total += 1
            if ok:
                n_x_valid += 1
        if moves_y:
            n_y_total += 1
            if ok:
                n_y_valid += 1
        if ok:
            stayed.append((dx, dy))

    n_total = len(probes)
    n_valid = len(stayed)
    frac = (n_valid / n_total) if n_total else 0.0

    # axis verdict: which axis's moves dominated the surviving probes.
    fx = (n_x_valid / n_x_total) if n_x_total else 0.0
    fy = (n_y_valid / n_y_total) if n_y_total else 0.0
    if abs(fx - fy) < 1e-9:
        axis = "both"
    elif fx > fy:
        axis = "x"
    else:
        axis = "y"

    if frac >= DONT_CARE_FRACTION:
        label = "dont_care"
        evidence = (
            f"{n_valid}/{n_total} perturbations stayed valid "
            f"(frac {frac:.2f} >= {DONT_CARE_FRACTION:.2f}): position is free design "
            f"space (x-free {fx:.2f}, y-free {fy:.2f}); declare FREE, store no coordinate"
        )
    elif n_valid <= FORCED_MAX_VALID:
        label = "missing_primitive"
        evidence = (
            f"only {n_valid}/{n_total} perturbations stayed valid "
            f"(<= {FORCED_MAX_VALID}): position is pinned -> a relation constrains it; "
            f"Phase C must discover the missing primitive (x-free {fx:.2f}, y-free {fy:.2f})"
        )
    else:
        label = "missing_primitive"
        evidence = (
            f"{n_valid}/{n_total} perturbations stayed valid "
            f"(frac {frac:.2f} below {DONT_CARE_FRACTION:.2f}): partially constrained "
            f"on the {axis if axis != 'both' else 'limiting'} axis "
            f"(x-free {fx:.2f}, y-free {fy:.2f}) -> missing primitive to pin it"
        )

    return PerturbResult(
        device=device,
        axis=axis,
        label=label,
        n_valid_perturbations=n_valid,
        n_total_perturbations=n_total,
        deltas_that_stayed_valid=stayed,
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# classify every device                                                       #
# --------------------------------------------------------------------------- #


def classify_all(
    positions: Dict[str, _Pos],
    sizes: Dict[str, _Size],
    intent: Any,
    *,
    deltas_um: Tuple[float, ...] = DEFAULT_DELTAS_UM,
    min_gap_um: float = DEFAULT_MIN_GAP_UM,
) -> Dict[str, PerturbResult]:
    """Classify every device in ``positions`` -> ``{name -> PerturbResult}``.

    Deterministic: devices are processed in sorted name order; the per-device probe
    set is fixed (no randomness). The MISSING-PRIMITIVE bucket of the result is the
    Phase-C worklist (devices needing a new relation); the DON'T-CARE bucket is the
    set to declare free.
    """
    return {
        name: classify_device(
            positions,
            sizes,
            intent,
            name,
            deltas_um=deltas_um,
            min_gap_um=min_gap_um,
        )
        for name in sorted(positions)
    }


# --------------------------------------------------------------------------- #
# summary                                                                      #
# --------------------------------------------------------------------------- #


def classification_summary(results: Dict[str, PerturbResult]) -> str:
    """A short report of the DON'T-CARE vs MISSING-PRIMITIVE split.

    Lists the count and members of each bucket. The MISSING-PRIMITIVE list tells
    Phase C exactly which devices need a new primitive discovered; the DON'T-CARE
    list tells it which to declare free. Deterministic (sorted members).
    """
    dont_care = sorted(n for n, r in results.items() if r.label == "dont_care")
    missing = sorted(n for n, r in results.items() if r.label == "missing_primitive")
    total = len(results)
    lines = [
        f"perturbation classification: {total} device(s)",
        f"  dont_care        : {len(dont_care)}  {dont_care}",
        f"  missing_primitive: {len(missing)}  {missing}",
    ]
    return "\n".join(lines)


__all__ = [
    "PerturbResult",
    "classify_device",
    "classify_all",
    "classification_summary",
    "reference_positions",
    "device_sizes",
    "DEFAULT_DELTAS_UM",
    "DONT_CARE_FRACTION",
    "FORCED_MAX_VALID",
]
